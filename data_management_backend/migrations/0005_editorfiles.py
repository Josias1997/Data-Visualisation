# Generated by Django 2.2.5 on 2020-02-27 12:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data_management_backend', '0004_file_title'),
    ]

    operations = [
        migrations.CreateModel(
            name='EditorFiles',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='ckeditor/files')),
            ],
        ),
    ]
