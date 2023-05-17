import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
# detectar a palma da mão e, depois,
#detectar os 21 pontos de referência na palma da mão.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
#confiança minima de detecção e rastreamento
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

tipIds = [4, 8, 12, 16, 20]

# Defina uma função para contar os dedos
#função para desenhar as conexões da mão
#desenhar pontos de referencia
def countFingers(image, hand_landmarks, handNo=0):
     # estamos obtendo as cordenadas da mão?
    if hand_landmarks:
        # Obtenha todos os pontos de referência da PRIMEIRA mão VISÍVEL
        landmarks = hand_landmarks[handNo].landmark
        # print(landmarks)

        # Conte os dedos
        fingers = []

        for lm_index in tipIds:
                # Obtenha os valores y da ponta e da parte inferior do dedo
                finger_tip_y = landmarks[lm_index].y 
                finger_bottom_y = landmarks[lm_index - 2].y

                # Verifique se ALGUM DEDO está ABERTO ou FECHADO
                if lm_index !=4:
                    if finger_tip_y < finger_bottom_y:
                        fingers.append(1)
                        print("DEDO com id ",lm_index," está Aberto")

                    if finger_tip_y > finger_bottom_y:
                        fingers.append(0)
                        print("DEDO com id ",lm_index," está Fechado")

        # print(fingers)
        totalFingers = fingers.count(1)

        # Exiba o texto
        text = f'Dedos: {totalFingers}'

        cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

# Defina uma função para 
def drawHandLanmarks(image, hand_landmarks):

    # Desenhar as conexões entre os pontos de referência
    # estamos obtendo as cordenadas da mão?
    if hand_landmarks:

      for landmarks in hand_landmarks:
         #img,   lista de pontos/corde  >algoritimo mediapip pra isso          
        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)


while True:
    success, image = cap.read()
     #0 inverte verticalmente 1 inverte horizon..
    image = cv2.flip(image, 1)
    
    # Detecte os pontos de referência das mãos 
    results = hands.process(image)

    # Obtenha a posição do ponto de referência do resultado processado
    #ponto de referencia
    #retorna a pos x,y e z sendo x e y ponto na tel
    #sendo z a distancia do ponto de ref da webcam
    hand_landmarks = results.multi_hand_landmarks

    # Desenhe os pontos de referência
    #chama função que desenha os ponto de referencia da mao
    #com base na image(video) e cordenadas da mao na tela
    drawHandLanmarks(image, hand_landmarks)

    # Obtenha a posição dos dedos da mão        
    countFingers(image, hand_landmarks)

    cv2.imshow("Controlador de Midia", image)

    # Saia da tela ao pressionar a barra de espaços
    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()
