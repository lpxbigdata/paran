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
        pickle.dump(map, open('small_data_test/mid/mapor.txt', 'wb+'))
        pickle.dump(maprev, open('small_data_test/mid/maprev.txt', 'wb+'))
        return nodenum

    def sort_data(self):
        mapor = pickle.load(open('small_data_test/mid/maprev.txt', 'rb'))
        originData = np.loadtxt(self.originDataPath, dtype='int')
        fout=open(self.sortedDataPath,'w')
        for i in range(originData.shape[0]):
            tmp = originData[i]
            l1=mapor.get(tmp[0])
            l2 = mapor.get(tmp[1])
            fout.write(str(l1)+" "+str(l2)+'\n')

    def to_blockmatrix(self):
        sortedData=np.loadtxt(self.sortedDataPath,dtype='int')
        # sortedData是源文件转换成映射的
        # sortedData.shape[0]代表行数
        first = True
        tar = []
        for i in range(sortedData.shape[0]):
            tmp = sortedData[i] #tmp是一个1*2的数组 [from to]
            if i == sortedData.shape[0]-1:
                tar.append(tmp[1])
                degree = len(tar)
                blocks = [[degree] for _ in range(self.basketnum)]
                for item in tar:
                    blocks[int(item / self.basketsize)].append(item)
                print('tar:%a  i:%d  blocks:%a'%(tar,i,blocks))
                for bas in range(self.basketnum):
                    if len(blocks[bas]) > 1:
                        pickle.dump(blocks[bas], open('small_data_test/mid/blocks_%d_%d' % (tmp[0], bas), 'wb'))
                        # 被存储到blocks_from_tar块号
                tar=[]
            elif sortedData[i+1,0] != sortedData[i,0]:
                tar.append(tmp[1])
                degree = len(tar)
                blocks = [[degree] for _ in range(self.basketnum)]# 不要浅拷贝！！！
                for item in tar:
                    blocksid=int(item / self.basketsize)
                    blocks[blocksid].append(item)
                print('tar:%a  i:%d  blocks:%a'%(tar,i,blocks))
                for bas in range(self.basketnum):
                    if len(blocks[bas]) > 1:
                        pickle.dump(blocks[bas], open('small_data_test/mid/blocks_%d_%d' % (tmp[0], bas), 'wb'))#第一个是from第二个是块
                tar=[]
            else:
                tar.append(tmp[1])

    def generate_top(self):
        for item in range(self.basketnum):
            r = [ 1.0 / (self.nodenum) for _ in range(self.basketsize)]
            f = open('small_data_test/rold/rold_%d' % item, 'wb')
            pickle.dump(r, f)
            f.close()
        while True:
            e = 0
            # item 块
            #print('self.nodenum:%d'%self.nodenum)
            for item in range(self.basketnum): #to的分块
                #一块里r_new值
                r_new = np.array([(1.0 - beta) / self.nodenum for _ in range(self.basketsize)])
                for src in range(self.nodenum): #src是from
                    if not os.path.exists('small_data_test/mid/blocks_%d_%d' % (src, item)):
                        continue
                    r_old = pickle.load(open('small_data_test/rold/rold_%d' % (int(src/self.basketsize)), 'rb'))
                    line = pickle.load(open('small_data_test/mid/blocks_%d_%d' % (src, item), 'rb'))
                    di = line[0]
                    destList = [nodes for nodes in line[1:]]
                    print('src:',src,'destList',destList)
                    for k in destList:
                        #print('k:%d  src除self.basketsize取余:%d'%(k,src % self.basketsize))
                        #r_new[k] += beta * r_old[src % self.basketsize] / di
                        r_new[k % self.basketsize ] += beta * r_old[src % self.basketsize] / di
                        print('item: %d ,r_new[%d] += beta * %f / di'%(item,k,r_old[src % self.basketsize]))
                f = open('small_data_test/rold/rnew_%d' % item, 'wb')
                pickle.dump(r_new, f)
                f.close()

                ro = pickle.load(open('small_data_test/rold/rold_%d' % item, 'rb'))
                #print('for e:r_new',r_new,'r_old:',r_old)
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
                temp=sorted(range(len(x)), key=lambda i: x[i], reverse=True)[:self.top]#前self.top的编号
                score=sorted(x,reverse=True)[:self.top];
                mapor = pickle.load(open('small_data_test/mid/mapor.txt', 'rb'))
                fout=open(self.resultDataPath,'wb')
                print('x%a'%x)
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
