from collections import deque
class DequeLRU():
    def __init__(self, maxsize):
        self.cache = deque()
        self.marks = {}
        # left MRU right LRU
        self.maxsize = maxsize
        self.total_ios = 0
        self.total_hits = 0
        self.total_pres = 0
        self.total_prehits = 0

    def _boost(self, lba):
        self.cache.remove(lba)
        self.cache.appendleft(lba)

    def __contains__(self, lba):
        return lba in self.cache

    def __repr__(self):
        return str(self.cache)

    def get_size(self):
        return len(self.cache)

    def get_last(self):
        return self.cache[-1]

    def get_first(self):
        return self.cache[0]

    def remove(self, lba):
        self.cache.remove(lba)
        return True

    def frict(self):
        return self.cache.popleft()

    def evict(self):
        return self.cache.pop()

    def full(self):
        return len(self.cache) == self.maxsize

    def push_back(self, lba):
        self.cache.append(lba)
        assert len(self.cache) <= self.maxsize
    # push the new, return the popped lba else None

    def push(self, lba, lbamark='n'):

        if lbamark == 'n':
            self.total_ios += 1
        if self.maxsize == 0:
            return None
        popped = None
        if lba in self.cache:

            self._boost(lba)

            if lbamark =='n':
                self.total_hits += 1

                if self.marks[lba] == 'p':
                    self.total_prehits += 1
                    self.marks[lba] = 'n'

            return None

        else:
            if lbamark == 'p':
                self.total_pres += 1
            if self.full():
                popped = self.cache.pop()
                poppedmark = self.marks.pop(popped)

            self.cache.appendleft(lba)
            self.marks[lba] = lbamark

            if popped:
                return popped, poppedmark
            else:
                return None

    def get_hit_rate(self):
        return self.total_hits / (self.total_ios + 1e-16)

    def get_prehit_rate(self):
        return self.total_prehits / (self.total_pres + 1e-16)

    def get_stats(self):
        return self.total_ios,self.total_pres,self.total_hits,self.total_prehits


class CacheTest():

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.cache = DequeLRU(maxsize=maxsize)

    def __contains__(self, lba):
        return lba in self.cache

    def __repr__(self):
        return self.cache.__repr__()

    #push normal io
    def push_normal(self,lba):
        return self.cache.push(lba=lba,lbamark='n')

    #push prefetch io
    def push_prefetch(self,lba,lazy_prefetch = False):
        if lazy_prefetch:
            if lba in self.cache:
                return None
            else:
                return self.cache.push(lba=lba,lbamark='p')
        else:
            return self.cache.push(lba=lba,lbamark='p')

    def get_hit_rate(self):
        return  self.cache.get_hit_rate()

    def get_prehit_rate(self):
        return  self.cache.get_prehit_rate()

    def get_stats(self):
        return  self.cache.get_stats()