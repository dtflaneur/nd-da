#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.cElementTree as ET
import pprint
import re
import codecs
import json
import collections
import pymongo
import re
from collections import defaultdict

import os
datadir = "data"
datafile = "mumbai.osm"
cal_data = os.path.join(datadir, datafile)

def count_tags(filename):
        """ function count_tags will parse through Mumbai dataset with ElementTree and count the number of unique elements to get an overview of the data and use pretty print to print the results. """
        tags = {}
        for event, elem in ET.iterparse(filename):
            if elem.tag in tags: 
                tags[elem.tag] += 1
            else:
                tags[elem.tag] = 1
        return tags
cal_tags = count_tags(cal_data)
pprint.pprint(cal_tags)

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

def key_type(element, keys):
    """function 'key_type', stores count of each of three tag categories in a dictionary: "lower", for tags that contain only lowercase letters and are valid,
"lower_colon", for otherwise valid tags with a colon in their names,"problemchars", for tags with problematic characters"""
    if element.tag == "tag":
        for tag in element.iter('tag'):
            k = tag.get('k')
            if lower.search(k):
                keys['lower'] += 1
            elif lower_colon.search(k):
                keys['lower_colon'] += 1
            elif problemchars.search(k):
                keys['problemchars'] += 1
            else:
                keys['other'] += 1
    return keys
        
def process_map(filename):
    """Returns a dictionary of tags & their counts"""
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return keys

cal_keys = process_map(cal_data)
pprint.pprint(cal_keys)


def process_map(filename):
    """Returns Unique contributers"""
    users = set()
    for _, element in ET.iterparse(filename):
        for e in element:
            if 'uid' in e.attrib:
                users.add(e.attrib['uid'])
    return users
users = process_map(cal_data)
len(users)

street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

expected = ["Road"]
mapping = {'Rd'   : 'Road',
           'Rd.'   : 'Road',
           'road'   : 'Road',
           'road No.' :"Road",
           'Road No.' :"Road",
           }
                   
def audit_street_name(street_name):
    """function searches the input string for the regex. If there is a match and it is not within the "expected" list, add the match as a key and add the string to the set."""
    m = street_type_re.search(street_name)
    if m:
        street = m.group()
        if street not in expected:
                    return street_name
                        
def is_street_name(tag):
    """function looks at the attribute k if k="addr:street" """
    return (tag.attrib['k'] == "addr:street")
        
def audit(elem):
    """function will return the list that match previous two functions. """
    #osm_file = open(osmfile, "r")
    street_name = defaultdict(set)
    name = ''
    #for event, elem in ET.iterparse(osm_file, events=("start",)):
    if elem.tag == "node" or elem.tag == "way":
                for tag in elem.iter("tag"):
                        if is_street_name(tag):
                                name = audit_street_name(tag.attrib['v'])
    return name

#cal_street_types = audit(cal_data)
#print "===============================cal_street_types",cal_street_types
#pprint.pprint(dict(cal_street_types))

def update_name(elem, mapping, regex):
    """function update_name takes the old name and update them with a better name """
    name = ''
    m = ''    
    name = audit(elem)
    if name:
            m = regex.search(name)
    if m:
        street_type = m.group()
        if street_type in mapping:
            name = re.sub(regex, mapping[street_type], name)

        if elem.tag == "node" or elem.tag == "way":
                for tag in elem.iter("tag"):
                        if  name and is_street_name(tag) and name!=tag.get('v'):
                                #print name
                                #print tag.get('v')
                                tag.attrib['v']=name
    return elem

#for street_type, ways in cal_street_types.iteritems():
#    for name in ways:
#        better_name = update_name(elem, mapping, street_type_re)
#        print name, "=>", better_name
                
def audit_zipcode(zipcode):
    """ check anomalies in zipcode"""
    twoDigits = zipcode[0:2]
    izip = ""
    if not twoDigits.isdigit():
        izip = zipcode
    elif twoDigits != 95:
        izip = zipcode
    return izip
    
def is_zipcode(tag):
                """function looks at the attribute k if k="addre:postcode" """
                return (tag.attrib['k'] == "addr:postcode")

def audit_zip(elem):
    """find zipcodes based on above 2 functions"""
    #osm_file = open(osmfile, "r")
    #invalid_zipcodes = defaultdict(set)
    izip = ""
    #for event, elem in ET.iterparse(osm_file, events=("start",)):
    if elem:
            if elem.tag == "node" or elem.tag == "way":
                    for tag in elem.iter("tag"):
                            if is_zipcode(tag):
                                    izip=audit_zipcode(tag.attrib['v'])
    return izip
#cal_zipcode = audit_zip(cal_data)

def update_zip(elem):
                """function update_zip takes the old zipcode and update them """
                zipcode = audit_zip(elem); #print "--------------------",zipcode
                testNum = re.findall('[a-zA-Z]*', zipcode); #print testNum
                if testNum:
                        testNum = testNum[0]
                testNum.strip()
                if testNum == "CA":
                        convertedZipcode = (re.findall(r'\d+', zipcode))
                        if convertedZipcode:
                                if convertedZipcode.__len__() == 2:
                                    zipcode = (re.findall(r'\d+', zipcode))[0] + "-" +(re.findall(r'\d+', zipcode))[1]
                        else:
                                    zipcode = (re.findall(r'\d+', zipcode))[0]
                                    
                if elem:
                        if elem.tag == "node" or elem.tag == "way":
                                for tag in elem.iter("tag"):
                                        if is_zipcode(tag):
                                        
                                                if elem.tag:
                                                        #print tag.get('v')
                                                        tag.attrib['v']=zipcode
                return elem				

#for street_type, ways in cal_zipcode.iteritems():
#    for name in ways:
#        better_name = update_zip(name)
#        print name, "=>", better_name

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
address_regex = re.compile(r'^addr\:')
street_regex = re.compile(r'^street')

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]


def shape_element(element):
    """To transform the data from XML to JSON, we should follow these rules:
       Process only 2 types of top level tags: "node" and "way"
       All attributes of "node" and "way" should be turned into regular key/value pairs, except: attributes in the CREATED array should be added under a key "created", attributes for latitude and longitude should be added to a "pos" array, for use in geospacial indexing. Make sure the values inside "pos" array are floats and not strings.
       If second level tag "k" value contains problematic characters, it should be ignored
       If second level tag "k" value starts with "addr:", it should be added to a dictionary "address"
       If second level tag "k" value does not start with "addr:", but contains ":", process it same as any other tag.
       If there is a second ":" that separates the type/direction of a street, ignore this tag"""
    node = {}
    address = {}
    if element:
			if element.tag == "node" or element.tag == "way" :
						node['type'] = element.tag
						
						# parsing through attributes
			for a in element.attrib:
				if a in CREATED:
					if 'created' not in node:
						node['created'] = {}
					node['created'][a] = element.get(a)
				elif a in ['lat', 'lon']:
					continue
				else:
					node[a] = element.get(a)
			if 'lat' in element.attrib and 'lon' in element.attrib:
				node['pos'] = [float(element.get('lat')), float(element.get('lon'))]

			# parse second-level tags for nodes
			for e in element:
				# parse second-level tags for ways and populate `node_refs`
				if e.tag == 'nd':
					if 'node_refs' not in node:
						node['node_refs'] = []
					if 'ref' in e.attrib:
						node['node_refs'].append(e.get('ref'))

				# throw out not-tag elements and elements without `k` or `v`
				if e.tag != 'tag' or 'k' not in e.attrib or 'v' not in e.attrib:
					continue
				key = e.get('k')
				val = e.get('v')

				# skip problematic characters
				if problemchars.search(key):
					continue

				# parse address k-v pairs
				elif address_regex.search(key):
					key = key.replace('addr:', '')
					address[key] = val
				# catch-all
				else:
					node[key] = val
			# compile address
			if len(address) > 0:
				node['address'] = {}
				street_full = None
				street_dict = {}
				street_format = ['prefix', 'name', 'type']
				# parse through address objects
				for key in address:
					val = address[key]
					if street_regex.search(key):
						if key == 'street':
							street_full = val
						elif 'street:' in key:
							street_dict[key.replace('street:', '')] = val
					else:
						node['address'][key] = val
				# assign street_full or fallback to compile street dict
				if street_full:
					node['address']['street'] = street_full
				elif len(street_dict) > 0:
					node['address']['street'] = ' '.join([street_dict[key] for key in street_format])
			return node
    else:
        return None


def process_map(file_in, pretty = False):
    """function to convert the file from XML into JSON."""
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, eleme in ET.iterparse(file_in):
                eleme = update_name(eleme, mapping, street_type_re)
                eleme = update_zip(eleme)
                el = shape_element(eleme)
                if el:
                        data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data
process_map(cal_data)

