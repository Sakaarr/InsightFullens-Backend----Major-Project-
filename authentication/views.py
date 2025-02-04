from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from drf_spectacular.utils import extend_schema, OpenApiExample
from .serializers import UserSerializer

@extend_schema(
    request=UserSerializer,
    responses={201: UserSerializer},
    examples=[
        OpenApiExample(
            name="Example Request",
            description="A sample request for registering a user",
            value={
                "username": "testuser",
                "email": "test@example.com",
                "password": "testpassword"
            },
            request_only=True
        )
    ]
)
class RegisterView(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@extend_schema(
    request=UserSerializer,
    responses={
        200: OpenApiExample(
            name="Example Response",
            description="Successful authentication response",
            value={
                "refresh": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "access": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        ),
        400: OpenApiExample(
            name="Error Response",
            description="Response for invalid credentials",
            value={"error": "Invalid Credentials"}
        )
    },
    examples=[
        OpenApiExample(
            name="Example Request",
            description="A sample login request",
            value={
                "username": "testuser",
                "password": "testpassword"
            },
            request_only=True
        )
    ]
)
class LoginView(APIView):
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        user = authenticate(username=username, password=password)
        if user:
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }, status=status.HTTP_200_OK)
        return Response({'error': 'Invalid Credentials'}, status=status.HTTP_400_BAD_REQUEST)
