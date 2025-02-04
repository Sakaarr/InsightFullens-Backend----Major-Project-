from django.urls import path
from .views import ProductSearchOrScrapeView

urlpatterns = [
    path('api/product/', ProductSearchOrScrapeView.as_view(), name='product-search-or-scrape'),
]
