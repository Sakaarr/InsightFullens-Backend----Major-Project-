import os,time
import csv
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.edge.service import Service as EdgeService
from .models import Product, ProductReview
from .serializers import ProductSerializer , ProductSearchInputSerializer
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


model_path = "C:/Users/DELL/Desktop/InsightFullens/insightfullens/product/fine_tuned"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)
def extract_aspect_sentiment(reviews):
    outputs = set()  # Store unique extracted aspects and sentiments
    for r in reviews:
        test_input = f"aspect sentiment extraction: {r}"
        input_ids = tokenizer(test_input, return_tensors="pt").input_ids

        # Generate output from the model
        output_ids = model.generate(input_ids, max_length=128)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.add(output)

    return list(outputs)
def init_driver():
    edge_options = Options()
    edge_options.add_argument("--disable-blink-features=AutomationControlled")
    edge_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    edge_options.add_argument("--disable-popup-blocking")
    edge_options.add_argument("--start-maximized")
    edge_driver_path = EdgeChromiumDriverManager().install()
    service = EdgeService(edge_driver_path)
    return webdriver.Edge(service=service, options=edge_options)

def scrape_product_reviews(product_link):
    driver = init_driver()
    driver.get(product_link)
    
    time.sleep(5)
    try:
        product_name = driver.find_element(By.ID, "productTitle").text.strip()
        reviews = []
        review_elements = driver.find_elements(By.CSS_SELECTOR, 'div[data-hook="review-collapsed"]')
        # for review_element in review_elements:
        #     review_text = review_div.text.strip()
        #     reviews.append(review_text)
        reviews = [review_element.text.strip() for review_element in review_elements]
    except Exception as e:
        product_name = None
        reviews = []
        print(f"Error: {e}")
    finally:
        driver.quit()
    return product_name, reviews

class ProductSearchOrScrapeView(APIView):
    @extend_schema(
        request=ProductSearchInputSerializer,  # Use the defined serializer
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
                    extracted_aspects = extract_aspect_sentiment(reviews)
                    for review_text in reviews:
                        ProductReview.objects.create(product=product, review_text=review_text)
                    return Response({"product_name": product_name, "reviews": reviews,"extracted_aspects": extracted_aspects}, status=status.HTTP_200_OK)
                else:
                    return Response({"message": "Failed to scrape product details."}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            try:
                product = Product.objects.get(name__icontains=user_input)
                serializer = ProductSerializer(product)
                return Response(serializer.data, status=status.HTTP_200_OK)
            except Product.DoesNotExist:
                csv_file_path = os.path.join(os.path.dirname(__file__), "products.csv")
                results = []
                try:
                    with open(csv_file_path, mode="r", encoding="utf-8") as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            if user_input.lower() in row["Product Name"].lower():
                                results.append({"Product Name": row["Product Name"], "Review": row["Product Review"]})
                    if results:
                        return Response({"products": results}, status=status.HTTP_200_OK)
                    else:
                        return Response({"message": "No matching products found."}, status=status.HTTP_404_NOT_FOUND)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"message": "Input cannot be empty."}, status=status.HTTP_400_BAD_REQUEST)
