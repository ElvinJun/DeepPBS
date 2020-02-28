from django.db import models

# Create your models here.
class Files(models.Model):
    id = models.AutoField(max_length=10, primary_key=True, verbose_name='id')
    file = models.FileField(upload_to='./files')
    def __unicode__(self):  # __str__ on Python 3
        return (self.id,self.file)

class Files_name(models.Model):
    id = models.AutoField(max_length=10, primary_key=True, verbose_name='id')
    name = models.CharField(max_length=10)
    files = models.ManyToManyField(Files, related_name='files')
    def __unicode__(self):  # __str__ on Python 3
        return (self.id,self.name,self.files)