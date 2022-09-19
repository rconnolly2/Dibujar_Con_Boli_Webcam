from asyncio.windows_events import NULL
import cv2
import numpy as np

webcam = cv2.VideoCapture(0)

def Redimensionar_Fotograma(fotograma, factor):
    height = fotograma.shape[0]
    width = fotograma.shape[1]
    height = height * factor
    height = round(height)
    width = width * factor
    width = round(width)
    return cv2.resize(fotograma, (width, height), interpolation=cv2.INTER_AREA)

def DetectarColor(mascaraazul, mascaranaranja, fotograma):
    #Miramos primero si en el fotograma hay azul si lo hay, convertimos a gris => Difuminamos => Lo convertimos en binario
    fotograma_mascara_azul = cv2.bitwise_and(fotograma, fotograma, mask=mascaraazul)
    fotograma_mascara_azul_gris = cv2.cvtColor(fotograma_mascara_azul, cv2.COLOR_BGR2GRAY)
    fotograma_mascara_azul_gris_difuminado = cv2.GaussianBlur(fotograma_mascara_azul_gris, (5, 5), 0)
    _, fotograma_mascara_azul_gris_difuminado_binario = cv2.threshold(fotograma_mascara_azul_gris_difuminado, 120, 255, cv2.THRESH_BINARY)

    #Miramos primero si en el fotograma hay naranja si lo hay, convertimos a gris => Difuminamos => Lo convertimos en binario
    fotograma_mascara_naranja = cv2.bitwise_and(fotograma, fotograma, mask=mascaranaranja)
    fotograma_mascara_naranja_gris = cv2.cvtColor(fotograma_mascara_naranja, cv2.COLOR_BGR2GRAY)
    fotograma_mascara_naranja_gris_difuminado = cv2.GaussianBlur(fotograma_mascara_naranja_gris, (5, 5), 0)
    _, fotograma_mascara_naranja_gris_difuminado_binario = cv2.threshold(fotograma_mascara_naranja_gris_difuminado, 120, 255, cv2.THRESH_BINARY)

    #Comprobamos que imagen esta en en negro el que no esta en negro totalmente es la imagen naranja si no azul que retornamos:
    color_detectado = ""
    if np.any(cv2.findNonZero(fotograma_mascara_azul_gris_difuminado_binario)) == None:
        color_detectado = "NARANJA"
        return fotograma_mascara_naranja_gris_difuminado_binario, color_detectado
    else:
        color_detectado = "AZUL"
        return fotograma_mascara_azul_gris_difuminado_binario, color_detectado


#Variables globales
lista_puntosxy_naranja = []
lista_puntosxy_azul = []
NARANJA = (25, 168, 255)
AZUL = (255, 247, 25)

while (webcam.isOpened()):
    ret, fotograma = webcam.read()
    fotograma_resize = Redimensionar_Fotograma(fotograma, 0.5)
    fotograma_HSV = cv2.cvtColor(fotograma_resize, cv2.COLOR_BGR2HSV)

    #Boli azul:
    blue_upper = np.array([173, 255, 255])
    blue_lower = np.array([34, 151, 182])

    #Lapiz naranja:
    orange_upper = np.array([88, 242, 255])
    orange_lower = np.array([0, 54, 156])


    mascara_azul = cv2.inRange(fotograma_HSV, blue_lower, blue_upper)
    mascara_naranja = cv2.inRange(fotograma_HSV, orange_lower, orange_upper)

    #Llamamos a la funcion Detectar color que manda las dos mascaras y nos deveulve el color detectado y el fotograma procesado lista para deteccion de contornos
    fotograma_mascara_gris_difuminado_binario, color_detectado = DetectarColor(mascara_azul, mascara_naranja, fotograma_resize)

     #Buscamos contornos
    contornos_boli, _ = cv2.findContours(fotograma_mascara_gris_difuminado_binario, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    lista_areas_contornos = []
    for contorno in contornos_boli:
        lista_areas_contornos.append(cv2.contourArea(contorno))
        area = cv2.contourArea(contorno)
        if area > 80:
            approx = cv2.approxPolyDP(contorno, 0.02*cv2.arcLength(contorno, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.circle(fotograma_resize, (x+w//2, y), 10, (255, 0, 0), thickness=2)
            #Comprobamos con que color estamos trabajando y añadimos cordenadas a la lista correcta
            if color_detectado == "NARANJA": 
                lista_puntosxy_naranja.append([x, y, w, h])
            else:
                lista_puntosxy_azul.append([x, y, w, h])
            break
    print(lista_areas_contornos)

    iterador = 0
    if not lista_areas_contornos == []:
        areamayor = max(lista_areas_contornos)
    for contornos in lista_areas_contornos:
        if contornos > 15 and contornos == areamayor:
           cv2.drawContours(fotograma_resize, contornos_boli, iterador, (255, 0, 0), 1)
           break
        else:
            iterador = iterador+1

    #Pintamos lista entera de posiciones de circulos naranjas
    for circulo in lista_puntosxy_naranja:
        x = circulo[0]
        y = circulo[1]
        w = circulo[2]
        h = circulo[3]
        cv2.circle(fotograma_resize, (x+w//2, y), 3, NARANJA, 7)

    #Pintamos lista entera de posiciones de circulos azul
    for circulo in lista_puntosxy_azul:
        x = circulo[0]
        y = circulo[1]
        w = circulo[2]
        h = circulo[3]
        cv2.circle(fotograma_resize, (x+w//2, y), 3, AZUL, 7)


    cv2.imshow("Dibujar Webcam", fotograma_resize)
    cv2.imshow("Mascara", fotograma_mascara_gris_difuminado_binario)
    waitkey = cv2.waitKey(50)
    if waitkey == ord("q"):
        break

cv2.destroyAllWindows()