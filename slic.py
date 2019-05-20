import cv2
import sys
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

class Cluster(object):
    cluster_index = 1

    def __init__(self, l, a, b, x, y):
        self.update(l, a, b, x, y)
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, l, a, b, x, y):
        self.l = (l)
        self.a = (a)
        self.b = (b)
        self.x = (x)
        self.y = (y)

    def down_index(self):
        Cluster.cluster_index -= 1

class SlicSuperpixels():
    #K값, M값을 받는다.
    def __init__(self, filename, K, m):
        #Clutser의 개수
        self.K = K

        #Distance를 구할 때 계산을 위한 값
        self.m = m

        self.image = io.imread(filename)
        self.image = color.rgb2lab(self.image)

        #data에 region을 저장할 것임.
        self.data = self.image;
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]

        #넓이
        self.N = self.image_width * self.image_height

        #S의 크기로 각 픽셀을 나눔. - distance를 구할 때와 나중에 labeling작업에 필요.
        self.S = (np.sqrt(self.N / self.K))
        self.centers = []

    def inner_cluster(self, x, y):
        return Cluster(self.data[x][y][0], #l
                       self.data[x][y][1], #a
                       self.data[x][y][2], #b
                       x, y)
    #Clusters를 얻는다

    def init_data(self):

        #클러스터와 distance를 초기화
        self.distances = sys.float_info.max*np.ones((self.image_height, self.image_width))
        self.clusters =  -1*np.ones((self.image_height, self.image_width))
        i = 0
        for x in range(int(self.image_height / self.S)):
            for y in range(int(self.image_width/self.S)):
                clutser_x = int(self.S/2 + x*self.S)
                cluster_y = int(self.S/2 + y*self.S)
                point = clutser_x, cluster_y
                nc = self.find_local_minimum(point)
                self.centers.append(self.inner_cluster(nc[0], nc[1]))
                #print(self.Kcluster, i)
                if i < self.K - 1:
                    i += 1
                else:
                    break
        self.result_color = [[0]*3 for i in range(len(self.centers))]
        #self.result_color2 = [[0] * 3 for i in range(len(self.centers))]
        """"
        #center 초기화
        for x in range(self.K, int(self.image_height - self.K/2), self.K):
            for y in range(self.K, int(self.image_width - self.K/2), self.K):
                point = x, y
                point = self.find_local_minimum(point)
                point_x = point[0]
                point_y = point[1]
                self.centers.append(self.inner_cluster(point_x, point_y))
        """""

    #image gradients compute
    #이미지의 3x3 인접 지역에서 최소 그라데이션 찾기
    def find_local_minimum(self, point):
        #point => (x, y) / point[0] => x, point[1] = y
        point_x = point[0]
        point_y = point[1]
        min_grad = sys.float_info.max
        local_minimum = point #this mean x, y

        for i in range(point_x-1, point_x+2):
            for j in range(point_y-1, point_y+2):
                #I is the lab vector corresponding to the pixel at position x, y
                I1 = self.data[i+1][j][0]#*0.11 + self.data[i+1][j][1]*0.59 + self.data[i+1][j][2]*0.3
                I2 = self.data[i][j+1][0]#*0.11 + self.data[i][j+1][1]*0.59 + self.data[i][j+1][2]*0.3
                I3 = self.data[i][j][0]#*0.11 + self.data[i][j][1]*0.59 + self.data[i][j][2]*0.3

                if np.sqrt(pow(I1 - I3, 2)) + np.sqrt(pow(I2 - I3, 2)) < min_grad:
                    min_grad = np.fabs(I1 - I3) + np.fabs(I2 - I3)
                    local_minimum = i, j

        return local_minimum


    #Distance measure
    def compute_dist(self, cluster, pixel):
        pixel_x = int(pixel[0])
        pixel_y = int(pixel[1])
        color = self.data[pixel_x][pixel_y]
        pixel_value = self.data[pixel_x][pixel_y]
        # compute Dlab
        Dc = np.sqrt(pow(cluster.l - color[0], 2) +
                     pow(cluster.a - color[1], 2) +
                     pow(cluster.b - color[2], 2))

        # compute Dxy
        Dp = np.sqrt(pow(cluster.x - pixel_x, 2) +
                     pow(cluster.y - pixel_y, 2))

        # Ds
        return Dc + (self.m/self.S)*Dp
        #return np.sqrt(pow(Dc / self.m, 2) + pow(Dp / self.S, 2))


    def genrate_superpixesl(self):
        Clustering_count = 5
        self.init_data()
        self.result_color = [[0] * 3 for i in range(len(self.centers))]
        for i in range(Clustering_count):
            #print("clustering count centers size: ", len(test.centers))
            #반복 계산
            for j in range(self.image_height):
                for k in range(self.image_width):
                    self.distances[j][k] = sys.float_info.max

            m = 0
            for center in self.centers:
                # 2Sx2S region
                for k in range(int(center.x - self.S), int(center.x + self.S)):
                    for l in range(int(center.y - self.S), int(center.y + self.S)):

                        if k >= 0 and k < self.image_height and l >=0 and l < self.image_width:
                            point = k, l
                            d = self.compute_dist(center, point)

                            if d < self.distances[k][l]:
                                self.distances[k][l] = d
                                self.clusters[k][l] = m

                if i == 0:
                    self.result_color[m] = (center.l, center.a, center.b)
                m += 1


            #plt.imshow(self.clusters)
            #plt.title(("Test: ", i))
            #plt.show()

            # centers 초기화

            for center in self.centers:
                center.update(0, 0, 0, 0, 0)
                #center.no = 0
                Cluster.cluster_index = 1

            for j in range(self.image_height):
                for k in range(self.image_width):
                    center_id = int(self.clusters[j][k])

                    if center_id != -1:
                        self.centers[center_id - 1] = self.inner_cluster(j, k)

                        #self.centers[center_id - 1].update(self.data[j][k][0],
                        #                                   self.data[j][k][1],
                        #                                   self.data[j][k][2],
                        #                                    j, k)
            # Normalize the clusters
            for center in self.centers:
                count = center.no
                #print(count)
                center.update(center.l / count,
                              center.a / count,
                              center.b / count,
                              center.x / count,
                              center.y / count)
    """""
    def create_connectivity(self):
        label = 0
        adjlabel = 0
        lims = int((self.image_width * self.image_height) / (int(len(self.centers))))

        dx4 = [-1, 0, 1, 0]
        dy4 = [0, -1, 0, 1]

        new_clusters = -1*np.ones((self.image_height, self.image_width))

        for i in range(self.image_height):
            for j in range(self.image_width):
                if new_clusters[i][j] == -1:
                    elements = []
                    elements.append([i, j])

                    for k in range(4):
                        x = elements[0][0] + dx4[k]
                        y = elements[0][1] + dy4[k]

                        if x >= 0 and x < self.image_height and y >=0 and y < self.image_width:
                            if new_clusters[x][y] >= 0:
                                adjlabel = new_clusters[x][y]

                    count = 1
                    for c in range(count):
                        for k in range(4):
                            x = elements[c][0] + dx4[k]
                            y = elements[c][1] + dy4[k]

                            if x >= 0 and x < self.image_height and y >= 0 and y < self.image_width:
                                if new_clusters[x][y] == -1 and self.clusters[i][j] == self.clusters[x][y]:
                                    elements.append([x, y])
                                    new_clusters[x][y] = label
                                    count += 1

                    if count <= (lims >> 2) :
                        for c in range(count):
                            new_clusters[elements[c][0]][elements[c][1]] = adjlabel
                        label -= 1
                    label += 1
    """""
    def inner_color(self):
        result = self.image
        result2 = self.image
        for x in range(self.image_height):
            for y in range(self.image_width):
                result[x][y] = self.result_color[int(self.clusters[x][y])]

        result = color.lab2rgb(result)
        plt.imshow(result)
        plt.show()

test = SlicSuperpixels("girl.jpg", 400, 60)

test.genrate_superpixesl()
print("Test.centers size", len(test.centers))
#test.create_connectivity()
plt.imshow(test.clusters)
plt.show()
test.inner_color()
