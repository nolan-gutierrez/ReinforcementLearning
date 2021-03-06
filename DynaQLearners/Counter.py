from numpy import random as r
class Counter:
    def __init__(self):
        self.counter = {}
    def getDict(self):
        return self.counter
    def initKeys(self,keys):
        for i in keys:
            self.keyValue(i,0)
    def setDict(self,dictionary):
        self.counter = dictionary
    def at(self,key):
        return self.counter[key]
    def keyValue(self,key,newValue):
        self.counter[key] = newValue
    def getKeys(self):
        return list(self.counter.keys())
    def getItems(self):
        return self.counter.items()
    def getItemsAsLists(self):
        keys = []
        values = []
        for k,v in self.getItems():
            keys.append(k)
            values.append(v)
        return keys,values
    def getValues(self):
        return list(self.counter.values())
    def getRandomKey(self):
        keys = self.counter.keys()
        return r.choice(keys)
    def argmax(self):
        #This function returns the best key  based on 
        #current values
        if len(self.getKeys()) == 0: return None
        bestKey = self.getKeys()[0]
        bestValue = self.counter[bestKey]
        if not any(self.getValues()):
            pick = r.choice(self.getKeys())
            return pick, self.counter[pick]
        for key,value in self.getItems():
            if value > bestValue: 
                bestValue = value
                bestKey = key
        return bestKey, bestValue
    def max(self):
        if len(self.getKeys()) == 0: return None
        bestKey = self.getKeys()[0]
        bestValue = self.counter[bestKey]
        for key,value in self.getItems():
            if value > bestValue: 
                bestValue = value
                bestKey = key
            return bestValue
    def mul(self,factor, copy = True):
        nc = Counter()
        if copy: nc.setDict(self.getDict().copy())
        else: nc = self
        for key, value in self.getItems():
            nc.keyValue(key, self.at(key) * factor)
        return nc

    def normalize(self, copy = True):
        total = 0
        for key, value in self.getItems():
            total += value
        return self.mul(1./total, copy)


