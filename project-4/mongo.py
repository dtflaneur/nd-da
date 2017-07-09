#!/usr/bin/env python
# -*- coding: utf-8 -*-

import signal
import subprocess
from pymongo import MongoClient
import os
import pprint

datadir = "data"
datafile = "mumbai.osm"
cal_data = os.path.join(datadir, datafile)

pro = subprocess.Popen('mongod')

db_name = 'openstreetmap'

# Connect to Mongo DB
client = MongoClient('localhost:27017')
db = client[db_name]

# Build mongoimport command
collection = cal_data[:cal_data.find('.')]
json_file = cal_data + '.json'

mongoimport_cmd = 'mongoimport -h 127.0.0.1:27017 ' + '--db ' + db_name + ' --collection ' + collection + ' --file ' + json_file

# Before importing, drop collection if it is already running 
if collection in db.collection_names():
    print 'Dropping collection: ' + collection
    db[collection].drop()
    
# Execute the command
print 'Executing: ' + mongoimport_cmd
subprocess.call(mongoimport_cmd.split())

mumbai = db[collection]

# File sizes
print 'The original OSM file is {} MB'.format(os.path.getsize(cal_data)/1.0e6) # convert from bytes to mb
print 'The JSON file is {} MB'.format(os.path.getsize(cal_data + ".json")/1.0e6) # convert from bytes to mb

# Number of documents
print 'Number of documents', mumbai.find().count()

# Number of unique users
print 'Number of unique users' ,len(mumbai.distinct('created.user'))

# Number of Nodes and Ways
print "Number of nodes:",mumbai.find({'type':'node'}).count()
print "Number of ways:",mumbai.find({'type':'way'}).count()

# Name of top 5 contributors
result = mumbai.aggregate( [
                                        { "$group" : {"_id" : "$created.user", 
                                        "count" : { "$sum" : 1} } },
                                        { "$sort" : {"count" : -1} }, 
                                        { "$limit" : 5 } ] )
print 'Name of top 5 contributors'
pprint.pprint(list(result))

# List of top 10 amenities in Mumbai
amenity = mumbai.aggregate([{'$match': {'amenity': {'$exists': 1}}}, \
                                {'$group': {'_id': '$amenity', \
                                            'count': {'$sum': 1}}}, \
                                {'$sort': {'count': -1}}, \
                                {'$limit': 10}])

print 'List of top 20 amenities in Mumbai'
pprint.pprint(list(amenity))

# List of top 5 Foods in Mumbai
cuisine = mumbai.aggregate([{"$match":{"amenity":{"$exists":1},
                                 "amenity":"restaurant",}},      
                      {"$group":{"_id":{"Food":"$cuisine"},
                                 "count":{"$sum":1}}},
                      {"$project":{"_id":0,
                                  "Food":"$_id.Food",
                                  "Count":"$count"}},
                      {"$sort":{"Count":-1}}, 
                      {"$limit":5}])
print 'List of top 5 Foods in Mumbai'
pprint.pprint(list(cuisine))

# List of top 10 post code in Mumbai
postcode = mumbai.aggregate( [ 
    { "$match" : { "address.postcode" : { "$exists" : 1} } }, 
    { "$group" : { "_id" : "$address.postcode", "count" : { "$sum" : 1} } },  
    { "$sort" : { "count" : -1}},
      {"$limit":10}] )
print 'List of top 10 post code in Mumbai'
pprint.pprint(list(postcode))

# Total users have unique post (post only one time)
users = mumbai.aggregate( [
    { "$group" : {"_id" : "$created.user", 
                "count" : { "$sum" : 1} } },
    { "$group" : {"_id" : "$count",
                "num_users": { "$sum" : 1} } },
    { "$sort" : {"_id" : 1} },
    { "$limit" : 1} ] )

print 'Total users have unique post (post only one time)',list(users)

# Top 5 building type by count
building = mumbai.aggregate([
       {'$match': {'building': { '$exists': 1}}}, 
        {'$group': {'_id': '$building',
                    'count': {'$sum': 1}}}, 
        {'$sort': {'count': -1}},
        {'$limit': 5}])
print 'Count by type of building'
pprint.pprint(list(building))

