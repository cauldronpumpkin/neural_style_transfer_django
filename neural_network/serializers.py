from rest_framework import serializers
from .models import NeuralNet


class NeuralNetSerializer(serializers.ModelSerializer):
    class Meta:
        model = NeuralNet
        fields = "__all__"
