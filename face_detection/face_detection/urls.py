from django.http import JsonResponse
from django.contrib import admin
from django.urls import path
from FLDPLDBM.views import signup_view, loginPage, logoutPage, landingPage, delete_embedding, FaceRecognitionAPI, home_page
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path("", home_page, name="home_page"),
    path('signup/', signup_view, name='signup'),
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
