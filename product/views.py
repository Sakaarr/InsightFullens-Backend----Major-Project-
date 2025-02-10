import os
import time
import csv
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, serializers
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.edge.service import Service as EdgeService
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

from .models import Product, ProductReview
from .serializers import ProductSerializer, ProductSearchInputSerializer
from drf_spectacular.utils import extend_schema

# Load model only once when module is imported
MODEL_PATH = "C:/Users/DELL/Desktop/InsightFullens/insightfullens/product/t5base"
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def deduplicate_prediction(prediction: str) -> str:
    """
    Removes duplicate aspect-sentiment pairs from predictions.
    
    Args:
        prediction (str): Raw prediction string containing aspect-sentiment pairs
        
    Returns:
        str: Deduplicated prediction string
    """
    # Split the prediction into individual parts and clean them
    parts = [part.strip() for part in prediction.split(",") if part.strip()]
    
    # Use a set to track seen parts while maintaining order
    seen = set()
    unique_parts = []
    
    for part in parts:
        # Only add if we haven't seen this exact aspect-sentiment pair before
        if part not in seen:
            seen.add(part)
            unique_parts.append(part)
    
    return ", ".join(unique_parts)

class WebDriverManager:
    """Manages WebDriver instances with proper cleanup"""
    def __init__(self):
        self.edge_options = self._configure_edge_options()
        self.driver_path = EdgeChromiumDriverManager().install()
        
    @staticmethod
    def _configure_edge_options():
        options = Options()
        options.add_argument("--headless")  # Run in headless mode
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("user-agent=Mozilla/5.0")
        return options
    
    def get_driver(self):
        service = EdgeService(self.driver_path)
        return webdriver.Edge(service=service, options=self.edge_options)

# Create a single WebDriverManager instance
driver_manager = WebDriverManager()

@lru_cache(maxsize=100)
def extract_aspect_sentiment_cached(review: str) -> str:
    """Cached version of aspect-sentiment analysis for individual reviews"""
    input_text = f"aspect-sentiment analysis: {review}"
    input_encoding = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_encoding["input_ids"],
            attention_mask=input_encoding["attention_mask"],
            max_length=50,
            num_beams=3,  # Reduced from 5 for better performance
            do_sample=True,
            temperature=0.7,
            top_k=50
        )

    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return deduplicate_prediction(prediction)

def extract_aspect_sentiment(reviews: List[str]) -> List[str]:
    """Parallel processing of reviews for aspect-sentiment analysis"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        predictions = list(executor.map(extract_aspect_sentiment_cached, reviews))
    return predictions

def scrape_product_reviews(product_link: str) -> Tuple[Optional[str], List[str]]:
    """Scrapes product name and reviews with improved error handling and timeouts"""
    driver = None
    try:
        driver = driver_manager.get_driver()
        driver.get(product_link)
        
        wait = WebDriverWait(driver, 10)  # 10 second timeout
        
        # Wait for product title to be present
        product_name = wait.until(
            EC.presence_of_element_located((By.ID, "productTitle"))
        ).text.strip()
        
        # Wait for reviews to be loaded
        review_elements = wait.until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, 'div[data-hook="review-collapsed"]')
            )
        )
        
        reviews = [element.text.strip() for element in review_elements if element.text.strip()]
        
        return product_name, reviews

    except TimeoutException:
        return None, ["Timeout while loading product page"]
    except NoSuchElementException:
        return None, ["Required elements not found on page"]
    except Exception as e:
        return None, [f"Error scraping reviews: {str(e)}"]
    finally:
        if driver:
            driver.quit()

class ProductSearchOrScrapeView(APIView):
    @extend_schema(
        request=ProductSearchInputSerializer,
        responses={200: ProductSerializer},
    )
    def post(self, request):
        user_input = request.data.get("input", "").strip()
        
        if not user_input:
            return Response(
                {"error": "Input is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if user_input.startswith(("http://", "https://")):
            return self._handle_url_query(user_input)
        return self._handle_search_query(user_input)

    def _handle_url_query(self, url: str) -> Response:
        try:
            product_name, reviews = scrape_product_reviews(url)
            
            if not product_name:
                return Response(
                    {"error": "Failed to scrape product details"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Process in batches of 10 reviews for memory efficiency
            batch_size = 10
            all_extracted_aspects = []
            for i in range(0, len(reviews), batch_size):
                batch = reviews[i:i + batch_size]
                all_extracted_aspects.extend(extract_aspect_sentiment(batch))

            # Create or update product and reviews
            product, _ = Product.objects.get_or_create(name=product_name)
            ProductReview.objects.bulk_create([
                ProductReview(product=product, review_text=review)
                for review in reviews
            ])

            summary_text = self._generate_summary(all_extracted_aspects)

            return Response({
                "product_name": product_name,
                "reviews": reviews,
                "extracted_aspects": all_extracted_aspects,
                "summary_text": summary_text
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"Failed to process URL: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _handle_search_query(self, query: str) -> Response:
        try:
            products = Product.objects.filter(name__icontains=query)
            if not products.exists():
                return Response(
                    {"message": "No matching products found."},
                    status=status.HTTP_404_NOT_FOUND
                )

            product_data = []
            for product in products:
                reviews = list(ProductReview.objects.filter(product=product)
                             .values_list("review_text", flat=True))
                
                extracted_aspects = (
                    extract_aspect_sentiment(reviews) if reviews else []
                )
                summary_text = (
                    self._generate_summary(extracted_aspects)
                    if reviews else "No reviews available for sentiment analysis."
                )

                product_info = ProductSerializer(product).data
                product_info.update({
                    "reviews": reviews,
                    "extracted_aspects": extracted_aspects,
                    "summary_text": summary_text,
                })
                product_data.append(product_info)

            return Response({"products": product_data}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"Search failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    # [Previous imports and initial setup remain the same until the _generate_summary method]

    @staticmethod
    def _generate_summary(extracted_aspects: List[str]) -> str:
        """
        Generates a summary from extracted aspects with sentiment counts.
        Handles all sentiment variations including extremely positive/negative and conflict.
        """
        aspect_sentiment_count = defaultdict(lambda: defaultdict(int))
        
        def normalize_sentiment(sentiment: str) -> str:
            """Normalize different sentiment variations to standard categories"""
            sentiment = sentiment.strip().lower()
            
            if "extremely positive" in sentiment:
                return "extremely positive"
            elif "extremely negative" in sentiment:
                return "extremely negative"
            elif "positive" in sentiment:
                return "positive"
            elif "negative" in sentiment:
                return "negative"
            elif "conflict" in sentiment:
                return "conflict"
            else:
                return "neutral"

        # Process aspects and count sentiments
        for aspect_string in extracted_aspects:
            for part in aspect_string.split(", "):
                if ":" in part:
                    try:
                        aspect, sentiment = part.rsplit(": ", 1)
                        aspect = aspect.strip().upper()
                        normalized_sentiment = normalize_sentiment(sentiment)
                        aspect_sentiment_count[aspect][normalized_sentiment] += 1
                    except Exception as e:
                        print(f"Error processing aspect-sentiment pair '{part}': {str(e)}")
                        continue

        # Classify aspects based on sentiment patterns
        def classify_aspect_sentiment(sentiments: dict) -> str:
            """
            Determine overall sentiment category based on sentiment counts
            Returns: 'very positive', 'positive', 'negative', 'very negative', 'mixed', or 'conflicting'
            """
            total_mentions = sum(sentiments.values())
            if total_mentions == 0:
                return None
                
            # Calculate percentages
            extremely_pos_pct = (sentiments["extremely positive"] / total_mentions) * 100
            pos_pct = (sentiments["positive"] / total_mentions) * 100
            neg_pct = (sentiments["negative"] / total_mentions) * 100
            extremely_neg_pct = (sentiments["extremely negative"] / total_mentions) * 100
            conflict_pct = (sentiments["conflict"] / total_mentions) * 100
            
            # Classification logic
            if conflict_pct > 20:  # If significant conflict mentions
                return "conflicting"
            elif extremely_pos_pct > 30:  # Strong positive sentiment
                return "very positive"
            elif extremely_neg_pct > 30:  # Strong negative sentiment
                return "very negative"
            elif (pos_pct + extremely_pos_pct) > (neg_pct + extremely_neg_pct + 10):  # Clear positive trend
                return "positive"
            elif (neg_pct + extremely_neg_pct) > (pos_pct + extremely_pos_pct + 10):  # Clear negative trend
                return "negative"
            elif total_mentions > 0:  # Mixed opinions
                return "mixed"
            
            return None

        # Organize aspects by sentiment category
        sentiment_categories = {
            "very positive": [],
            "positive": [],
            "negative": [],
            "very negative": [],
            "mixed": [],
            "conflicting": []
        }

        for aspect, sentiments in aspect_sentiment_count.items():
            category = classify_aspect_sentiment(sentiments)
            if category:
                sentiment_categories[category].append(aspect)

        # Generate detailed summary
        summary_parts = []
        
        if sentiment_categories["very positive"]:
            summary_parts.append(
                f"Users were extremely enthusiastic about the {', '.join(sentiment_categories['very positive'])}."
            )
            
        if sentiment_categories["positive"]:
            summary_parts.append(
                f"Many users found the {', '.join(sentiment_categories['positive'])} to be good."
            )
            
        if sentiment_categories["mixed"]:
            summary_parts.append(
                f"Opinions on the {', '.join(sentiment_categories['mixed'])} were mixed."
            )
            
        if sentiment_categories["conflicting"]:
            summary_parts.append(
                f"There were conflicting views about the {', '.join(sentiment_categories['conflicting'])}."
            )
            
        if sentiment_categories["negative"]:
            summary_parts.append(
                f"Users reported issues with the {', '.join(sentiment_categories['negative'])}."
            )
            
        if sentiment_categories["very negative"]:
            summary_parts.append(
                f"Users expressed strong concerns about the {', '.join(sentiment_categories['very negative'])}."
            )

        return " ".join(summary_parts) if summary_parts else "No clear sentiment patterns found."

# [Rest of the code remains the same]
class ReviewInputSerializer(serializers.Serializer):
    reviews = serializers.ListField(
        child=serializers.CharField(),
        help_text="A list of review texts to analyze."
    )


class AspectSentimentOutputSerializer(serializers.Serializer):
    predictions = serializers.ListField(
        child=serializers.CharField(),
        help_text="A list of aspect-sentiment predictions."
    )
    
class AspectSentimentAnalysisView(APIView):
    @extend_schema(
        request=ReviewInputSerializer,
        responses={200: AspectSentimentOutputSerializer},
    )
    def post(self, request, *args, **kwargs):
        serializer = ReviewInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        reviews = serializer.validated_data["reviews"]
        predictions = extract_aspect_sentiment(reviews)
        return Response({"predictions": predictions}, status=status.HTTP_200_OK)