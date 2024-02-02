from django.urls import path
from . import views
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ImageViewSet

router = DefaultRouter()
router.register(r'index', ImageViewSet, basename='index')

urlpatterns = [
    path("",views.index,name="index"),
    path('api/', include(router.urls)),
    # ... your other urlpatterns ...
]
