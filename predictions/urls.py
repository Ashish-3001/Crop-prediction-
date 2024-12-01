from django.urls import path
from .views import predict_view,Home, predict_view_1, predict_view_2, predict_view_3, predict_view_4

urlpatterns = [
    path('', Home, name='home'),
    path('home/', Home, name='home'),
    path('model1/', predict_view, name='predict'),
    path('model2/', predict_view_1, name='predict1'),
    path('model3/', predict_view_2, name='predict2'),
    path('model4/', predict_view_3, name='predict3'),
    path('model5/', predict_view_4, name='predict4'),
]