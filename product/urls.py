from django.urls import path
from .views import ProductSearchOrScrapeView,AspectSentimentAnalysisView,ProductDetailView

urlpatterns = [
    path('api/product/', ProductSearchOrScrapeView.as_view(), name='product-search-or-scrape'),
    path('test/', AspectSentimentAnalysisView.as_view(), name='test-sentiment-analysis'),
    path('api/product/<int:product_id>/', ProductDetailView.as_view(), name='product-detail'),
]
