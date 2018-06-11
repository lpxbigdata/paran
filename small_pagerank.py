import numpy as np
import pickle
import math
import os

class pagerank:

    def __init__(self,top,beta,basketsize):
        self.originDataPath = 'data/test_input1.txt'
        self.sortedDataPath = 'data/mappedtest_input1.txt'
        self.resultDataPath = 'data/test_result.txt'
        self.top=top
        self.beta=beta
        self.nodes = self.read_file()
        self.basketsize=basketsize
        self.basketnum=int(math.ceil(float(len(self.nodes)) / basketsize))
        print(
            self.basketnum
        )
        self.nodenum=self.sort_node()
        print('nodenum%d'%self.nodenum)
        self.sort_data()
        self.to_blockmatrix()
        self.generate_top()

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
        pickle.dump(map, open('data/mid/mapor.txt', 'wb+'))
        pickle.dump(maprev, open('data/mid/maprev.txt', 'wb+'))
        return nodenum

    def sort_data(self):
        mapor = pickle.load(open('data/mid/maprev.txt', 'rb'))
        originData = np.loadtxt(self.originDataPath, dtype='int')
        dist=[]
        for i in range(originData.shape[0]):
            tmp = originData[i]
            l1=mapor.get(tmp[0])
            l2 = mapor.get(tmp[1])
            dist.append([l1,l2])
        np.savetxt(self.sortedDataPath,dist)

    def to_blockmatrix(self):
        sortedData=np.loadtxt(self.sortedDataPath,dtype='int')
        # sortedData是源文件转换成映射的
        # sortedData.shape[0]代表行数
        dist = []
        for i in range(sortedData.shape[0]):
            tmp = sortedData[i] #tmp是一个1*2的数组 [from to]
            if i == sortedData.shape[0]-1:
                dist.append(tmp[1])
                degree = len(dist)
                blocks = [[degree] for _ in range(self.basketnum)]
                for item in dist:
                    blocks[int(item / self.basketsize)].append(item)
                print('dist:%a  i:%d  blocks:%a'%(dist,i,blocks))
                for bas in range(self.basketnum):
                    if len(blocks[bas]) > 1:
                        np.save(('data/mid/blocks_%d_%d' % (tmp[0], bas)),blocks[bas])
                dist=[]
            elif sortedData[i+1,0] != sortedData[i,0]:
                dist.append(tmp[1])
                degree = len(dist)
                blocks = [[degree] for _ in range(self.basketnum)]
                for item in dist:
                    blocksid=int(item / self.basketsize)
                    blocks[blocksid].append(item)
                print('dist:%a  i:%d  blocks:%a'%(dist,i,blocks))
                for bas in range(self.basketnum):
                    if len(blocks[bas]) > 1:
                        np.save(('data/mid/blocks_%d_%d' % (tmp[0], bas)), blocks[bas])
                dist=[]
            else:
                dist.append(tmp[1])

    def generate_top(self):
        for item in range(self.basketnum):
            r = [ 1.0 / (self.nodenum) for _ in range(self.basketsize)]
            np.save('data/oldr/oldr_%d.npy' % item, r)
        e = 1
        while e > 1e-7:
            e = 0
            # item 块
            #print('self.nodenum:%d'%self.nodenum)
            for item in range(self.basketnum): #to的分块
                #一块里r_new值
                r_new = np.array([(1.0 - beta) / self.nodenum for _ in range(self.basketsize)])
                for src in range(self.nodenum): #src是from
                    if not os.path.exists('data/mid/blocks_%d_%d' % (src, item)):
                        continue
                    r_old = np.load('data/oldr/oldr_%d.npy' % (int(src/self.basketsize)))
                    line = np.load('data/mid/blocks_%d_%d' % (src, item))
                    di = line[0]
                    destList = [nodes for nodes in line[1:]]
                    for k in destList:
                        r_new[k % self.basketsize ] += beta * r_old[src % self.basketsize] / di
                np.save('data/newr/newr_%d.npy' % item, r_new)
                ro = np.load('data/oldr/oldr_%d.npy' % item)
                #print('for e:r_new',r_new,'r_old:',r_old)
                e += np.linalg.norm((np.array(r_new) - np.array(ro)), ord=1)        # L1 norm
            print('e%f'%e)
            for i in range(self.basketnum):
                rn = np.load('data/newr/newr_%d.npy' % i)
                np.save('data/oldr/oldr_%d.npy' % i, rn)

        # print result
        x = []
        for i in range(self.basketnum):
            r = np.load('data/newr/newr_%d.npy' % i)
            for i in r:
                x.append(i)
                x = x[:self.nodenum]
        temp=sorted(range(len(x)), key=lambda i: x[i], reverse=True)[:self.top]#前self.top的编号
        score=sorted(x,reverse=True)[:self.top];
        mapor = pickle.load(open('data/mid/mapor.txt', 'rb'))
        fout=open(self.resultDataPath,'wb')
        print('x%a'%x)
        print('self.top%d'%self.top)
        re=[]
        for i in range(self.top):
            l1=mapor.get(temp[i]) # nodeid
            re.append([int(l1),score[i]])
            print('nodeid: %d   score[%d]: %f'%(l1,i,score[i]))
        np.savetxt("data/result.txt",re, fmt="%d %f")
        fout.close()


if __name__ == '__main__':
    top = 3
    beta = 0.85
    basketsize= 2
    p = pagerank(top, beta,basketsize)
