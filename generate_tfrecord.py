# Kütüphaneleri ekleme
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# Object detection klasöründe olduğumuz için bütün gerekenlere erişmek için path i belirliyoruz
sys.path.append("..")

# Modülleri ekleme
from utils import label_map_util
from utils import visualization_utils as vis_util

# Kullandığımız nesne tanıma modülünün klasörünü belirliyoruz
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'test.jpg'

# Üzerinde çalıştığımız dizini alıyoruz
CWD_PATH = os.getcwd()

# Nesne tanıma için kullanacağımız eğitilmiş modelimizi içeren inference_graph dosyasının dizinini belirtiyoruz.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Etiket haritamızın dizinini belirtiyoruz.
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Fotoğrafımızın dizinini belirtiyoruz.
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Sınıf sayımızı belirtiyoruz. Ben de dört tane sınıf olduğu için dört yazdım.
NUM_CLASSES = 4

# Etiket haritasını yüklüyoruz.
# Bu sayede modelimiz örnek olarak 4 sayısını tahmin olarak döndürdüğünde bu sayının ceket’e karşılık geldiğini bileceğiz.
# Burada biz dahili fayda fonksiyonlarını kullanıyoruz, fakat integer değerlerini string karşılığına çeviren herhangi bir sözlük #de kullanılabilir.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Eğittiğimiz tensorflow modelimizi yüklüyoruz.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Nesne algılama sınıflandırıcısı için giriş ve çıkış tensörlerini (yani verileri) tanımlama
# Giriş değerlerimiz fotoğraflamızı oluyor.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')


# Çıkış tensörleri algılama kutuları, puanlar ve sınıflardır.
# Burada tanımlama kutularını belirliyoruz.
# Her kutu, görüntünün belirli bir nesnenin algılandığı bölümünü temsil eder
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

#Burada modelin her bir nesneyi tanıma skor değerlerini belirliyoruz. 
# Her bir skor değeri modelimizin nesneyi ne kadar yüksek bir oranda tanıdığını temsil ediyor ve bu skorlar nesnenin etiketi #ve tanımlama kutuları ile beraber gözükecek.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Tanınan nesnelerin sayısını belirliyoruz. Tanınan nesne sayısını bilebileceğiz bu sayede.
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Resimleri opencv kullanarak yükleme ve resmi [1, None, None, 3] boyutlarına göre genişlet.
# Yani her bir satırdaki öğenin rgb değerleri olduğu tek sütunlu bir dizi oluştur.
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)

# Modeli input olarak resim ile çalıştırarak asıl algılamayı gerçekleştirme.
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Algılamanın sonuçlarını çiz, algılama kutularını ve algılama sonucunu gösterme.

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.80)

# Sonuçlarla beraber resmi gösterme.
cv2.imshow('Object detector', image)

# Resmi kapamak için herhangi bir tuşa basma.
cv2.waitKey(0)

# Herşeyi yok edip temizleme.
cv2.destroyAllWindows()
