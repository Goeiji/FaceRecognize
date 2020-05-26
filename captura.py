import cv2

n =input("Digite um número para visualizar a foto: ")
image_path = "fotos/" + n + ".jpg"
classificador = cv2.CascadeClassifier("original.xml")

#id = input("Digite o seu identificador: ")
#scale = 30

img = cv2.imread(image_path)
largura = 570
altura = 700

imgResize = cv2.resize(img, (largura, altura))
gray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)

facesDetectadas = classificador.detectMultiScale(gray,
                                                scaleFactor=1.3,
                                                minSize=(200, 200))
for(x, y, l, a) in facesDetectadas:
   cv2.rectangle(imgResize, (x, y), (x + l, y + a), (99, 253, 163), 2)

cv2.imshow('image', imgResize)
cv2.waitKey(0)
cv2.destroyAllWindows()