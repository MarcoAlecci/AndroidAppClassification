import datetime
import redis
import json

class RedisClient:

    # Client to interact with a Redis Server
    client     = None
    
    # Project Key
    projectKey = None

    # List of useful Keys
    popKey     = None
    resultsKey = None
    errorKey   = None

    def __init__(self, host, port, db, password, projectKey):
        # Client
        self.client     = redis.Redis(host=host, port=port, db=db, password=password)
        # Main Key
        self.projectKey = projectKey
        # List of elements to be analyzed
        self.popKey     = projectKey + ".pop"
        # List with results
        self.resultsKey = projectKey + ".result"
        # List of elements who generated an error
        self.errorKey   = projectKey + ".error"

    # Function to get the size of a Redis hash or list.
    def getSize(self, redisKey):
        # Check if the redisKey is a set or list
        if self.client.type(redisKey) == b'hash':
            return self.client.hlen(redisKey)
        elif self.client.type(redisKey) == b'list':
            return self.client.llen(redisKey)
        else:
            return None
        
    # Function to print current situation.
    def printStatus(self):
        # Print current date and time
        print("⏲️ Current Date and Time: {}".format(datetime.datetime.now()))

        # Define the keys for which size will be printed
        keys = [self.popKey, self.resultsKey, self.errorKey]

        # Print size for each key
        print("\n📐 Size")
        for key in keys:
            print("- {:<45} : {}".format(key, self.getSize(key)))

    # Delete one List/Set
    def deleteOne(self, redisKey):
        self.client.delete(redisKey)
        self.printStatus()

    # Delete all three Sets
    def deleteAll(self):
        # Define the keys for which size will be printed
        keys = [self.popKey, self.resultsKey, self.errorKey]
        
        # Delete
        for key in keys:
            self.client.delete(key)
            print("- {:<40} : {}".format(key, self.getSize(key)))

    # Load Pop List
    def loadPopList(self, values):
        # Push the values  to the pop List
        for v in values:
            self.client.rpush(self.popKey, v)
        self.printStatus()

    # Copy from one Key to another
    def copyRedisHashSets(self, sourceKey, destinationKey):
        # Get all fields and values from the source key
        values = self.client.hgetall(sourceKey)
        
        # Copy fields and values to the destination key
        for field, value in values.items():
            self.client.hset(destinationKey, field, value)
        
        # Print the number of elements in the destination key
        numElements = self.client.hlen(destinationKey)
        print(f"Number of elements in {destinationKey}: {numElements}")

    # Method to get data (Json Object) from an HSET in REDIS using a specific KEY and sha256 as key to access
    def downloadJsonData(self, redisKey, sha256):
        try:
            jsonDataString = self.client.hget(redisKey, sha256)
            if jsonDataString is not None:
                data = json.loads(jsonDataString)
                return data
            else:
                return None
        except Exception as e:
            print("--- ⚠️ An error occurred:", e)
            return None
        
    # Method to get data (String) from an HSET in REDIS using a specific KEY and sha256 as key to access
    def downloadString(self, redisKey, sha256):
        try:
            dataString = self.client.hget(redisKey, sha256).decode('utf-8')
            return dataString  
        except Exception as e:
            print("--- ⚠️ An error occurred:", e)
            return None