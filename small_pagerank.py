import numpy as np
import pickle
import math
import os

class pagerank:

    def __init__(self,top,beta,basketsize):
        self.originDataPath = 'small_data_test/test_input1.txt'
        self.sortedDataPath = 'small_data_test/mappedtest_input1.txt'
        self.resultDataPath = 'small_data_test/test_result.txt'
        self.top=top
        self.beta=beta
        self.basketsize = basketsize
        self.nodes = self.read_file()
        self.basketnum=int(math.ceil(float(len(self.nodes)) / basketsize))
        self.nodenum=self.sort_node()
        self.map_data()
        self.to_blockmatrix()
        self.generate_top()

    def read_file(self):
        originData = np.loadtxt(self.originDataPath,dtype='int')
        nodes = np.hstack((originData[:, 0],originData[:, 1]))
        nodes = np.unique(nodes)
        nodes.sort()
        return nodes

    def sort_node(self):
        nodenum = len(self.nodes)
        index = [i for i in range(nodenum)]
        map = dict(zip(index, self.nodes))
        maprev=dict(zip(self.nodes, index))
        np.savez("small_data_test/mid/mapor.txt",map)
        np.savez("small_data_test/mid/maprev.txt", maprev)
        return nodenum

    def map_data(self):
        mapor = np.load("small_data_test/mid/maprev.txt")
        originData = np.loadtxt(self.originDataPath, dtype='int')
        dist=[]
        for i in range(originData.shape[0]):
            tmp = originData[i]
            l1=mapor.get(tmp[0])
            l2 = mapor.get(tmp[1])
            dist.append([l1,l2])
        np.savetxt(self.sortedDataPath ,dist)

    def to_blockmatrix(self):
        sortedData=np.loadtxt(self.sortedDataPath,dtype='int')
        dist = []
        for i in range(sortedData.shape[0]):
            tmp = sortedData[i] 
            if i == sortedData.shape[0]-1:
                dist.append(tmp[1])
                degree = len(dist)
                blocks = [[degree] for _ in range(self.basketnum)]
                for item in dist:
                    blocks[int(item / self.basketsize)].append(item)
                for bas in range(self.basketnum):
                    if len(blocks[bas]) > 1:
                        pickle.dump(blocks[bas], open('small_data_test/mid/blocks_%d_%d' % (tmp[0], bas), 'wb'))
                dist=[]
            elif sortedData[i+1,0] != sortedData[i,0]:
                dist.append(tmp[1])
                degree = len(dist)
                blocks = [[degree] for _ in range(self.basketnum)]
                for item in dist:
                    blocksid=int(item / self.basketsize)
                    blocks[blocksid].append(item)
                for bas in range(self.basketnum):
                    if len(blocks[bas]) > 1:
                        pickle.dump(blocks[bas], open('small_data_test/mid/blocks_%d_%d' % (tmp[0], bas), 'wb'))
                dist=[]
            else:
                dist.append(tmp[1])

    def generate_top(self):
        for item in range(self.basketnum):
            r = [ 1.0 / (self.nodenum) for _ in range(self.basketsize)]
            f = open('small_data_test/rold/rold_%d' % item, 'wb')
            pickle.dump(r, f)
            f.close()
        while True:
            e = 0
            for item in range(self.basketnum):
                r_new = np.array([(1.0 - beta) / self.nodenum for _ in range(self.basketsize)])
                for src in range(self.nodenum):
                    if not os.path.exists('small_data_test/mid/blocks_%d_%d' % (src, item)):
                        continue
                    r_old = pickle.load(open('small_data_test/rold/rold_%d' % (int(src/self.basketsize)), 'rb'))
                    line = pickle.load(open('small_data_test/mid/blocks_%d_%d' % (src, item), 'rb'))
                    di = line[0]
                    destList = [nodes for nodes in line[1:]]
                    for k in destList:
                        r_new[k % self.basketsize ] += beta * r_old[src % self.basketsize] / di
                        print('item: %d ,r_new[%d] += beta * %f / di'%(item,k,r_old[src % self.basketsize]))
                f = open('small_data_test/rold/rnew_%d' % item, 'wb')
                pickle.dump(r_new, f)
                f.close()

                ro = pickle.load(open('small_data_test/rold/rold_%d' % item, 'rb'))
                e += np.linalg.norm((np.array(r_new) - np.array(ro)), ord=1)        # L1 norm
            print('e%f'%e)

            if e < 1e-6:
                # print result
                x = []
                for i in range(self.basketnum):
                    r = pickle.load(open('small_data_test/rold/rnew_%d' % i, 'rb'))
                    for i in r:
                        x.append(i)
                        x = x[:self.nodenum]
                temp=sorted(range(len(x)), key=lambda i: x[i], reverse=True)[:self.top]#old self.top number
                score=sorted(x,reverse=True)[:self.top]
                mapor = pickle.load(open('small_data_test/mid/mapor.txt', 'rb'))
                fout=open(self.resultDataPath,'wb')
                print(x)
                print('self.top%d'%self.top)
                for i in range(self.top):
                    l1=mapor.get(temp[i]) # nodeid
                    print('nodeid: %d   score[%d]: %f'%(l1,i,score[i]))
                fout.close()
                return

            for i in range(self.basketnum):
                rn = pickle.load(open('small_data_test/rold/rnew_%d' % i, 'rb'))
                pickle.dump(rn, open('small_data_test/rold/rold_%d' % i, 'wb'))


if __name__ == '__main__':
    top = 3
    beta = 0.85
    basketsize= 2
    p = pagerank(top, beta,basketsize)
