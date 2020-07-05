from django.db import models

# Create your models here.


class NeuralNet(models.Model):
    main_image = models.FileField(blank=False, null=False)
    style_image = models.FileField(blank=False, null=False)
