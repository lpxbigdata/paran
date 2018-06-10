import numpy as np
import pickle
import math

class pagerank:

    def __init__(self,top,beta,basketsize):
        self.originDataPath = 'data/WikiData.txt'
        self.sortedDataPath = 'data/WikiDataSorted.txt'
        self.top=top
        self.beta=beta
        self.nodes = self.read_file()
        self.basketsize=basketsize
        self.basketnum=int(math.ceil(float(len(self.nodes)) / basketsize))
        print(
            self.basketnum
        )
        self.sort_data()
        self.to_blockmatrix()

    def read_file(self):
        originData = np.loadtxt(self.originDataPath,dtype='int')
        nodes = np.hstack((originData[:, 0],originData[:, 1]))
        nodes = np.unique(nodes)
        return nodes

    def sort_node(self):
        nodenum = len(self.nodes)
        index = [i for i in range(nodenum)]
        map = dict(zip(index, self.nodes))
        maprev=dict(zip(self.nodes, index))
        pickle.dump(map, open('mid/mapor.txt', 'wb+'))
        pickle.dump(maprev, open('mid/maprev.txt', 'wb+'))

    def sort_data(self):
        mapor = pickle.load(open('mid/maprev.txt', 'rb'))
        originData = np.loadtxt(self.originDataPath, dtype='int')
        fout=open(self.sortedDataPath,'w')
        for i in range(originData.shape[0]):
            tmp = originData[i]
            l1=mapor.get(tmp[0])
            l2 = mapor.get(tmp[1])
            fout.write(str(l1)+" "+str(l2)+'\n')

    def to_blockmatrix(self):
        sortedData=np.loadtxt(self.sortedDataPath,dtype='int')
        first = True
        tar = []
        for i in range(sortedData.shape[0]):
            tmp = sortedData[i]
            if i == sortedData.shape[0]-1:
                tar.append(tmp[1])
                degree = len(tar)
                blocks = [[degree]] * self.basketnum
                for item in tar:
                    blocks[int(item / self.basketsize)].append(item)
                for bas in range(self.basketnum):
                    if len(blocks[bas]) > 1:
                        pickle.dump(blocks[bas], open('mid/blocks_%d_%d' % (tmp[0], bas), 'wb'))
                tar=[]
            elif sortedData[i+1,0] != sortedData[i,0]:
                tar.append(tmp[1])
                degree = len(tar)
                blocks = [[degree]] * self.basketnum
                for item in tar:
                    blocks[int(item / self.basketsize)].append(item)
                for bas in range(self.basketnum):
                    if len(blocks[bas]) > 1:
                        pickle.dump(blocks[bas], open('mid/blocks_%d_%d' % (tmp[0], bas), 'wb'))
                tar=[]
            else:
                tar.append(tmp[1])


if __name__ == '__main__':
    top = 100
    beta = 0.85
    basketsize=500
    p = pagerank(top, beta,basketsize)
