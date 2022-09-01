import streamlit as st
import numpy as np
import torch
import torchvision
from PIL import Image
from types import SimpleNamespace # простой класс, где можно прописать параметры
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn

output_path = 'C:\\PycharmProjects\\ganstylestream\\venv\\'

text1 = 'C:\\PycharmProjects\\ganstylestream\\venv\\MarieAntoinette.jpg'
text2 = 'C:\\PycharmProjects\\ganstylestream\\venv\\stylevangog.jpg'

st.write('Рассмотрим работу PyTorch')
uploaded_img = st.file_uploader("Ниже загрузите изображение для преобразования:", type=['jpg', 'jpeg', 'png'])
if uploaded_img is not None:
    img_array = np.array(Image.open(uploaded_img))
    st.image(img_array, use_column_width='auto', caption=f'Загруженное изображение {uploaded_img.name}')
    ppp = Image.fromarray(img_array)
    ppp.save(output_path + uploaded_img.name)
    text1 = output_path + uploaded_img.name

style_img = st.file_uploader("Ниже загрузите изображение, которое выбираете в качестве стиля:", type=['jpg', 'jpeg', 'png'])
if style_img is not None:
    style_array = np.array(Image.open(style_img))
    st.image(style_array, use_column_width='auto', caption=f'Загруженное изображение {style_img.name}')
    ggg = Image.fromarray(style_array)
    ggg.save(output_path + style_img.name)
    text2 = output_path + style_img.name


totalstep = st.slider('Сколько шагов итераций хотите задать для работы нейронки?', 0, 600, 100)
stepsave  = st.slider('Через сколько итераций сохранять полученное изображение?', 0,200,10)

config = SimpleNamespace() # Создаем базовый класс пространства имен
#config.content = 'C:\\PycharmProjects\\ganstylestream\\venv\\LudovikXVI.jpg' # наша основная картинка
config.content = text1
#config.style = 'C:\\PycharmProjects\\ganstylestream\\venv\\stylevangog.jpg' # наша стилизованная картинка
config.style = text2
config.maxSize = 400 # максимально допустимый размер изображения
config.totalStep = totalstep # общее количество шагов за эпоху
config.step = 10 # шаг
config.sampleStep = stepsave # шаг для сохранения образца
config.styleWeight = 100 #вес на стиль
config.lr = .003

# Проверка, если GPU включен
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PretrainedNet(nn.Module):
    def __init__(self):
        # Инициализируем нашу модель
        super(PretrainedNet, self).__init__()
        self.select = [0, 5, 7, 10, 15]  # те слои, через которые мы будем пропускать свое изображение
        self.pretrainedNet = models.vgg19(pretrained=True).to(device)  # подгружаем предобученную сетку

    def forward(self, x):
        features = []  # Извлекаем по индексам, которые мы прописали выше, feature map из сетки
        output = x
        for layerIndex in range(len(self.pretrainedNet.features)):
            output = self.pretrainedNet.features[layerIndex](output)
            if layerIndex in self.select:
                features.append(output)
        return features


def load_image(image_path, transform=None, maxSize=None, shape=None):
    # Загружаем изображение
    image = Image.open(image_path)

    # Если указан максимальный размер, то меняем размер нашего изображения
    if maxSize:
        scale = maxSize / max(image.size)  # задаем масштаб для преобразования размера
        size = np.array(image.size) * scale  # масштабированный размер
        image = image.resize(size.astype(int), Image.ANTIALIAS)  # преобразуем

    # Если указана форма изображением, меняем форму
    if shape:
        image = image.resize(shape, Image.LANCZOS)

    # Если указаны методы трансформирования, то применяем его
    if transform:
        image = transform(image).unsqueeze(0)  # трансформировали + вытянули до батча

    return image.to(device)

# Методы трансформирования изображения
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                            std=(0.229, 0.224, 0.225))])

# Загружаем оригинал и стиль для картинок, применив нужные методы
content = load_image(config.content, transform, maxSize=config.maxSize)
style = load_image(config.style, transform, shape=[content.size(2), content.size(3)])

# Создаем место под тензор для конечной картинки, указываем, что дифференцируем
target = content.clone().requires_grad_(True)

model = PretrainedNet().eval() # для использования весов предобученной сетки переводим ее в режим eval
optimizer = torch.optim.Adam([target], lr=0.1) # наша цель - не менять веса сетки, а менять изображение-тензор, поэтому указываем его в качестве параметра
contentCriteria = nn.MSELoss()

start_proc = st.button("Нажмите для запуска процесса смшения стилей")
if start_proc :
    for step in range(config.totalStep):
            # Для каждого из изображений извлекаем feature map
        targetFeatures = model.forward(target)
        contentFeatures = model.forward(content)
        styleFeatures = model.forward(style)
        styleLoss = 0
        contentLoss = 0

        for f1, f2, f3 in zip(targetFeatures, contentFeatures, styleFeatures):
            # Вычисляем потери для оригинала и конечной картинки
            contentLoss += contentCriteria(f1, f2)
            # print(contentLoss)
            # Меняем форму сверточных feature maps. Приводим к формату (количество каналов, ширина*высота)
            _, c, h, w = f1.size()  # пропускаем batch
            f1 = f1.reshape(c, h * w).to(device)
            f3 = f3.reshape(c, h * w).to(device)

            # Находим матрицу Грама для конечной и стиля
            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())

            # Потери для стиля и конечной картинки
            kf1 = 1 / (4 * (len(f1) * len(f3)) ** 2)
            kf2 = 1 / 4 * (len(f1) * len(f3)) ** 2
            kf3 = 1 / (c * w * h)
            styleLoss += contentCriteria(f1, f3) * kf2
        # Прописываем конечную функцию потерь
        loss = styleLoss + contentLoss
        # print(betta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % config.step == 0:
            print('Шаг [{}/{}], Ошибка для оригинала: {:.4f}, Ошибка для стиля: {}'
                  .format(step + 1, config.totalStep, contentLoss.item(), styleLoss.item()))

        if (step + 1) % config.sampleStep == 0:  # сохраняем нашу картинку
            img = target.clone().squeeze()  # создаем место под тензор
            img = img.clamp_(0, 1)  # оставить значения, попадающие в диапазон между 0,1
            torchvision.utils.save_image(img, output_path + 'output-{}.jpg'.format(step + 1))
            saved_img = Image.open(output_path + 'output-{}.jpg'.format(step + 1))
            st.image(saved_img, output_format='auto')

#styled_img = target.cpu().detach().numpy()[0].transpose(1,2,0)

