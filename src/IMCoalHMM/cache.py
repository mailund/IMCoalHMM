"""A cache table."""

class Cache(object):
    """A table for caching objects"""

    def __init__(self, cleanup_size=2000):
        self.cleanup_size = cleanup_size
        self.last_access = 0
        self.table = {}

    def __setitem__(self, key, value):
        self.table[key] = (value, self.last_access)
        self.last_access += 1
        if len(self.table) > self.cleanup_size:
            self.cleanup()
        return value

    def __getitem__(self, key):
        value, _ = self.table[key]
        self.table[key] = (value, self.last_access)
        self.last_access += 1
        return value

    def __contains__(self, key):
        if key not in self.table:
            return False
        # update last access
        value, _ = self.table[key]
        self.table[key] = (value, self.last_access)
        self.last_access += 1
        return True

    def cleanup(self):
        items = self.table.items()
        reordered = [(access,key,value) for key,(value,access) in items]
        reordered.sort(reverse=True)
        last_accessed = reordered[:self.cleanup_size]
        self.last_access = 0
        self.table = {}
        for access,key,value in last_accessed:
            self.table[key] = (value, self.last_access)
            self.last_access += 1