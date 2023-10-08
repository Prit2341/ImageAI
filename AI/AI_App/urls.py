from django.urls import path
from .views import render_page

urlpatterns = [
    path('', render_page, name='home'),
]
