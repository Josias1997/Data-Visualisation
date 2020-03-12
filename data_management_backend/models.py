from django.db import models
from django.contrib.auth.models import User
import secrets

# Create your models here.

class File(models.Model):
	title = models.CharField(max_length=100, default='default title')
	file = models.FileField(upload_to='files/')

	def __str__(self):
		return self.title



class Token(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='tokens')
	string = models.CharField(max_length=255)

	def __str__(self):
		return self.string

	def generateToken(self):
		self.string = secrets.token_urlsafe(16)


class PlotFile(models.Model):
	file = models.ImageField(upload_to='plot/files')


