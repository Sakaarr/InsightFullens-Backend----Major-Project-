from rest_framework import serializers
from .models import Product, ProductReview

class ProductReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductReview
        fields = ['id', 'review_text']

class ProductSerializer(serializers.ModelSerializer):
    reviews = ProductReviewSerializer(many=True, read_only=True)

    class Meta:
        model = Product
        fields = ['id', 'name', 'reviews']
        
class ProductSearchInputSerializer(serializers.Serializer):
    input = serializers.CharField(help_text="Enter a product name or URL")
