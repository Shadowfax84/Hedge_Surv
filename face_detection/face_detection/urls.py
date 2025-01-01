from django.http import JsonResponse
from django.contrib import admin
from django.urls import path
from FLDPLDBM.views import signup_view, loginPage, logoutPage, landingPage, delete_embedding, FaceRecognitionAPI
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', signup_view, name='signup'),  # Signup page as root path
    path('login/', loginPage, name='login'),
    path('logout/', logoutPage, name='logout'),
    path('landing/', landingPage, name='landing'),
    path('delete_embedding/', delete_embedding, name='delete_embedding'),
    path('recognition/<str:action>/',
         FaceRecognitionAPI.as_view(), name='recognition-api'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)


def start_fl_client_view(request):
    if request.method == "POST":
        start_flower_client()
        return JsonResponse({"message": "Flower client started!"})
    return JsonResponse({"error": "Invalid request method"}, status=400)
