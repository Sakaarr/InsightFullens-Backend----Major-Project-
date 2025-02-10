from django.urls import path
from .views import ProductSearchOrScrapeView,AspectSentimentAnalysisView

urlpatterns = [
    path('api/product/', ProductSearchOrScrapeView.as_view(), name='product-search-or-scrape'),
    path('test/', AspectSentimentAnalysisView.as_view(), name='test-sentiment-analysis'),
]
