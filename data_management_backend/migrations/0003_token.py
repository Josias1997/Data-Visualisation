# Generated by Django 2.2.4 on 2019-09-27 05:20

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('data_management_backend', '0002_auto_20190925_1311'),
    ]

    operations = [
        migrations.CreateModel(
            name='Token',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('string', models.CharField(max_length=255)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='tokens', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
