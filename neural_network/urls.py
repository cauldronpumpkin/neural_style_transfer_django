from django.urls import path

from django.contrib import admin

from .views import FileUploadView

from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    path('', FileUploadView.as_view(), name="hello"),
]

urlpatterns = format_suffix_patterns(urlpatterns)
