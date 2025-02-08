# import os,time
# import csv
# from collections import defaultdict
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.edge.options import Options
# from webdriver_manager.microsoft import EdgeChromiumDriverManager
# from selenium.webdriver.edge.service import Service as EdgeService
# from .models import Product, ProductReview
# from rest_framework import status, serializers
# from .serializers import ProductSerializer , ProductSearchInputSerializer
# from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes
# from transformers import T5ForConditionalGeneration, T5Tokenizer,BartTokenizer, BartForConditionalGeneration
# import torch
# from drf_spectacular.utils import extend_schema, OpenApiExample


# model_path = "C:/Users/DELL/Desktop/InsightFullens/insightfullens/product/t5base"
# model = T5ForConditionalGeneration.from_pretrained(model_path)
# tokenizer = T5Tokenizer.from_pretrained(model_path)
# model.eval()
# def extract_aspect_sentiment(reviews):
#     # outputs = set()  # Store unique extracted aspects and sentiments
#     # for r in reviews:
#     #     test_input = f"aspect sentiment extraction: {r}"
#     #     input_ids = tokenizer(test_input, return_tensors="pt").input_ids

#     #     # Generate output from the model
#     #     output_ids = model.generate(input_ids, max_length=128)
#     #     output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     #     outputs.add(output)

#     # return list(outputs)
   
#     """Generate predictions for a list of input review texts."""
#     predictions = []
#     for review in reviews:
#         input_text = f"aspect-sentiment analysis: {review}"
#         input_encoding = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
#         input_ids = input_encoding["input_ids"]
#         attention_mask = input_encoding["attention_mask"]

#         with torch.no_grad():
#             output_ids = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_length=50,  # Increase length to capture details
#                 num_beams=5, 
#                 do_sample=True,# Beam search to improve quality
#                 temperature=0.7,  # Lower temp to keep responses focused
#                 top_k=50 
#                 # Limit randomness in word choices
#             )

#         prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         prediction = deduplicate_prediction(prediction)
#         predictions.append(prediction)
    
#     return predictions
# def init_driver():
#     edge_options = Options()
#     edge_options.add_argument("--disable-blink-features=AutomationControlled")
#     edge_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
#     edge_options.add_argument("--disable-popup-blocking")
#     edge_options.add_argument("--start-maximized")
#     edge_driver_path = EdgeChromiumDriverManager().install()
#     service = EdgeService(edge_driver_path)
#     return webdriver.Edge(service=service, options=edge_options)

# def scrape_product_reviews(product_link):
#     driver = init_driver()
#     driver.get(product_link)
    
#     time.sleep(5)
#     try:
#         product_name = driver.find_element(By.ID, "productTitle").text.strip()
#         reviews = []
#         review_elements = driver.find_elements(By.CSS_SELECTOR, 'div[data-hook="review-collapsed"]')
#         # for review_element in review_elements:
#         #     review_text = review_div.text.strip()
#         #     reviews.append(review_text)
#         reviews = [review_element.text.strip() for review_element in review_elements]
#     except Exception as e:
#         product_name = None
#         reviews = []
#         print(f"Error: {e}")
#     finally:
#         driver.quit()
#     return product_name, reviews

# class ProductSearchOrScrapeView(APIView):
#     @extend_schema(
#         request=ProductSearchInputSerializer,  # Use the defined serializer
#         responses={200: ProductSerializer},
#     )
#     def post(self, request):
#         user_input = request.data.get("input", "").strip()

#         def is_url(input_string):
#             return input_string.startswith("http://") or input_string.startswith("https://")

#         if is_url(user_input):
#             try:
#                 product_name, reviews = scrape_product_reviews(user_input)
#                 if product_name:
#                     product, _ = Product.objects.get_or_create(name=product_name)
#                     extracted_aspects = extract_aspect_sentiment(reviews)
#                     extracted_aspects_text = " ".join(extracted_aspects)
#                     aspect_sentiment_count = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})

#                     # Process each extracted aspect sentiment
#                     for aspect_string in extracted_aspects:
#                         parts = aspect_string.split(", ")
#                         for part in parts:
#                             if ":" in part:
#                                 aspect, sentiment = part.rsplit(": ", 1)
#                                 aspect = aspect.strip().upper()  # Capitalizing aspect
#                                 aspect_sentiment_count[aspect][sentiment.strip()] += 1

#                     # Group aspects based on sentiment
#                     positive_aspects = [f"**{aspect}**" for aspect, counts in aspect_sentiment_count.items() if counts["positive"] > 0 and counts["negative"] == 0]
#                     negative_aspects = [f"**{aspect}**" for aspect, counts in aspect_sentiment_count.items() if counts["negative"] > 0 and counts["positive"] == 0]
#                     mixed_aspects = [f"**{aspect}**" for aspect, counts in aspect_sentiment_count.items() if counts["positive"] > 0 and counts["negative"] > 0]

#                     # Constructing the final summary
#                     summary_parts = []

#                     if positive_aspects:
#                         summary_parts.append(f"Many users found the {', '.join(positive_aspects)} of the product to be good.")
#                     if mixed_aspects:
#                         summary_parts.append(f"Opinions on the {', '.join(mixed_aspects)} are mixed; some users liked them, while others found them lacking.")
#                     if negative_aspects:
#                         summary_parts.append(f"Users reported issues with the {', '.join(negative_aspects)}.")

#                     # Generate final summary
#                     summary_text = " ".join(summary_parts)
#                     print(summary_text)

#                     for review_text in reviews:
#                         ProductReview.objects.create(product=product, review_text=review_text)
#                     return Response({
#                         "product_name": product_name,
#                         "reviews": reviews,
#                         "extracted_aspects": extracted_aspects,
#                         "summary_text": summary_text
#                     }, status=status.HTTP_200_OK)
#                 else:
#                     return Response({"message": "Failed to scrape product details."}, status=status.HTTP_400_BAD_REQUEST)
#             except Exception as e:
#                 return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#         else:
#             try:
#                 product = Product.objects.get(name__icontains=user_input)
#                 reviews_qs = ProductReview.objects.filter(product=product)
#                 reviews = list(reviews_qs.values_list("review_text", flat=True))
#                 if reviews:
#                     extracted_aspects = extract_aspect_sentiment(reviews)
#                     aspect_sentiment_count = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})

#                     for aspect_string in extracted_aspects:
#                         parts = aspect_string.split(", ")
#                         for part in parts:
#                             if ":" in part:
#                                 aspect, sentiment = part.rsplit(": ", 1)
#                                 aspect = aspect.strip().upper()
#                                 aspect_sentiment_count[aspect][sentiment.strip()] += 1

#                     positive_aspects = [f"**{aspect}**" for aspect, counts in aspect_sentiment_count.items() if counts["positive"] > 0 and counts["negative"] == 0]
#                     negative_aspects = [f"**{aspect}**" for aspect, counts in aspect_sentiment_count.items() if counts["negative"] > 0 and counts["positive"] == 0]
#                     mixed_aspects = [f"**{aspect}**" for aspect, counts in aspect_sentiment_count.items() if counts["positive"] > 0 and counts["negative"] > 0]

#                     summary_parts = []
#                     if positive_aspects:
#                         summary_parts.append(f"Many users found the {', '.join(positive_aspects)} of the product to be good.")
#                     if mixed_aspects:
#                         summary_parts.append(f"Opinions on the {', '.join(mixed_aspects)} are mixed; some users liked them, while others found them lacking.")
#                     if negative_aspects:
#                         summary_parts.append(f"Users reported issues with the {', '.join(negative_aspects)}.")
#                     summary_text = " ".join(summary_parts)
#                 else:
#                     extracted_aspects = []
#                     summary_text = "No reviews available for sentiment analysis."

#                 serializer = ProductSerializer(product)
#                 response_data = serializer.data
#                 response_data.update({
#                     "reviews": reviews,
#                     "extracted_aspects": extracted_aspects,
#                     "summary_text": summary_text,
#                 })
#                 return Response(response_data, status=status.HTTP_200_OK)
#             except Product.DoesNotExist:
#                 csv_file_path = os.path.join(os.path.dirname(__file__), "products.csv")
#                 results = []
#                 try:
#                     with open(csv_file_path, mode="r", encoding="utf-8") as file:
#                         reader = csv.DictReader(file)
#                         results = [row for row in reader if user_input.lower() in row["Product Name"].lower()]
#                         if results:
#                             return Response({
#                                 "products": results,
#                             }, status=status.HTTP_200_OK)
#                         else:
#                             return Response({"message": "No matching products found."}, status=status.HTTP_404_NOT_FOUND)
#                 except Exception as e:
#                     return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#             return Response({"message": "Input cannot be empty."}, status=status.HTTP_400_BAD_REQUEST)


# class ReviewInputSerializer(serializers.Serializer):
#     reviews = serializers.ListField(
#         child=serializers.CharField(),
#         help_text="A list of review texts to analyze."
#     )

# # Output serializer: Returns a list of predictions.
# class AspectSentimentOutputSerializer(serializers.Serializer):
#     predictions = serializers.ListField(
#         child=serializers.CharField(),
#         help_text="A list of aspect-sentiment predictions corresponding to the input reviews."
#     )
    
# def deduplicate_prediction(prediction: str) -> str:
#     """
#     Removes duplicate aspect-sentiment pairs from the prediction string.
    
#     Assumes that pairs are separated by commas. For example:
#         Input: "service: positive, food: negative, service: positive, food: negative"
#         Output: "service: positive, food: negative"
#     """
#     # Split on comma and strip extra whitespace.
#     parts = [part.strip() for part in prediction.split(",") if part.strip()]
    
#     # Remove duplicates while preserving order.
#     unique_parts = []
#     for part in parts:
#         if part not in unique_parts:
#             unique_parts.append(part)
    
#     # Join back into a single string.
#     return ", ".join(unique_parts)
    
# @extend_schema(
#     request=ReviewInputSerializer,
#     responses={200: AspectSentimentOutputSerializer},
#     examples=[
#         OpenApiExample(
#             'Example Request',
#             value={
#                 "reviews": [
#                     "The service was great, but the food was cold.",
#                     "I loved the ambiance but the price was too high."
#                 ]
#             }
#         ),
#         OpenApiExample(
#             'Example Response',
#             value={
#                 "predictions": [
#                     "Extracted aspects and sentiments for review 1",
#                     "Extracted aspects and sentiments for review 2"
#                 ]
#             }
#         )
#     ]
# )
# class AspectSentimentAnalysisView(APIView):
#     """
#     POST API endpoint for aspect sentiment analysis.
    
#     This endpoint accepts a JSON payload with a list of reviews and returns predictions
#     after processing each review using the extract_aspect_sentiment function.
#     """
#     def post(self, request, *args, **kwargs):
#         # Validate the incoming request data.
#         serializer = ReviewInputSerializer(data=request.data)
#         serializer.is_valid(raise_exception=True)
#         reviews = serializer.validated_data["reviews"]

#         # Get predictions by calling the extraction function.
#         predictions = extract_aspect_sentiment(reviews)

#         # Return the predictions in a JSON response.
#         return Response({"predictions": predictions}, status=status.HTTP_200_OK)
import os
import time
import csv
from collections import defaultdict
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, serializers
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.edge.service import Service as EdgeService
from .models import Product, ProductReview
from .serializers import ProductSerializer, ProductSearchInputSerializer
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes, OpenApiExample
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from rest_framework.decorators import api_view
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from .tasks import extract_aspect_sentiment_task
# Load pre-trained model
model_path = "C:/Users/DELL/Desktop/InsightFullens/insightfullens/product/t5base"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)
model.eval()


def extract_aspect_sentiment(reviews):
    """Generates aspect-sentiment analysis for a list of reviews."""
    predictions = []
    for review in reviews:
        input_text = f"aspect-sentiment analysis: {review}"
        input_encoding = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_encoding["input_ids"],
                attention_mask=input_encoding["attention_mask"],
                max_length=50,
                num_beams=5,
                do_sample=True,
                temperature=0.7,
                top_k=50
            )

        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prediction = deduplicate_prediction(prediction)
        predictions.append(prediction)

    return predictions


def init_driver():
    """Initializes and returns the Edge WebDriver."""
    edge_options = Options()
    edge_options.add_argument("--disable-blink-features=AutomationControlled")
    edge_options.add_argument("user-agent=Mozilla/5.0")
    edge_options.add_argument("--disable-popup-blocking")
    edge_options.add_argument("--start-maximized")
    edge_driver_path = EdgeChromiumDriverManager().install()
    service = EdgeService(edge_driver_path)
    return webdriver.Edge(service=service, options=edge_options)


def scrape_product_reviews(product_link):
    """Scrapes product name and reviews from the given product link."""
    driver = init_driver()
    driver.get(product_link)
    time.sleep(5)

    try:
        product_name = driver.find_element(By.ID, "productTitle").text.strip()
        review_elements = driver.find_elements(By.CSS_SELECTOR, 'div[data-hook="review-collapsed"]')
        reviews = [review_element.text.strip() for review_element in review_elements]
    except Exception as e:
        product_name, reviews = None, []
        print(f"Error: {e}")
    finally:
        driver.quit()

    return product_name, reviews


class ProductSearchOrScrapeView(APIView):
    @extend_schema(
        request=ProductSearchInputSerializer,
        responses={200: ProductSerializer},
    )
    def post(self, request):
        user_input = request.data.get("input", "").strip()

        def is_url(input_string):
            return input_string.startswith("http://") or input_string.startswith("https://")

        if is_url(user_input):
            try:
                product_name, reviews = scrape_product_reviews(user_input)
                if product_name:
                    product, _ = Product.objects.get_or_create(name=product_name)
                    extracted_aspects = extract_aspect_sentiment_tasks.delay(reviews)

                    aspect_sentiment_count = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
                    for aspect_string in extracted_aspects:
                        parts = aspect_string.split(", ")
                        for part in parts:
                            if ":" in part:
                                aspect, sentiment = part.rsplit(": ", 1)
                                aspect = aspect.strip().upper()
                                aspect_sentiment_count[aspect][sentiment.strip()] += 1

                    positive_aspects = [f"{aspect}" for aspect, counts in aspect_sentiment_count.items() if counts["positive"] > 0 and counts["negative"] == 0]
                    negative_aspects = [f"{aspect}" for aspect, counts in aspect_sentiment_count.items() if counts["negative"] > 0 and counts["positive"] == 0]
                    mixed_aspects = [f"{aspect}" for aspect, counts in aspect_sentiment_count.items() if counts["positive"] > 0 and counts["negative"] > 0]

                    summary_parts = []
                    if positive_aspects:
                        summary_parts.append(f"Many users found the {', '.join(positive_aspects)} to be good.")
                    if mixed_aspects:
                        summary_parts.append(f"Opinions on the {', '.join(mixed_aspects)} are mixed.")
                    if negative_aspects:
                        summary_parts.append(f"Users reported issues with the {', '.join(negative_aspects)}.")
                    summary_text = " ".join(summary_parts)

                    for review_text in reviews:
                        ProductReview.objects.create(product=product, review_text=review_text)

                    return Response({
                        "product_name": product_name,
                        "reviews": reviews,
                        "extracted_aspects": extracted_aspects,
                        "summary_text": summary_text
                    }, status=status.HTTP_200_OK)
                else:
                    return Response({"message": "Failed to scrape product details."}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            try:
                products = Product.objects.filter(name__icontains=user_input)
                if not products.exists():
                    return Response({"message": "No matching products found."}, status=status.HTTP_404_NOT_FOUND)

                product_data = []
                for product in products:
                    reviews_qs = ProductReview.objects.filter(product=product)
                    reviews = list(reviews_qs.values_list("review_text", flat=True))
                    extracted_aspects =extract_aspect_sentiment_tasks.delay(reviews) if reviews else []
                    summary_text = "No reviews available for sentiment analysis." if not reviews else " ".join(extracted_aspects)

                    serializer = ProductSerializer(product)
                    product_info = serializer.data
                    product_info.update({
                        "reviews": reviews,
                        "extracted_aspects": extracted_aspects,
                        "summary_text": summary_text,
                    })
                    product_data.append(product_info)

                return Response({"products": product_data}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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


def deduplicate_prediction(prediction: str) -> str:
    """Removes duplicate aspect-sentiment pairs from predictions."""
    parts = [part.strip() for part in prediction.split(",") if part.strip()]
    unique_parts = []
    for part in parts:
        if part not in unique_parts:
            unique_parts.append(part)
    return ", ".join(unique_parts)


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