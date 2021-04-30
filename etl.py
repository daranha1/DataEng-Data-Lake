############################################################################################################################ 
#  Diana Aranha
#  Project 4: Data Lake
#  Udacity Nanodegree : Data Engineering

# This python script :
# 1. reads two Udacity datasets located on AWS S3 -- song_data  and log_data
# 2. transforms it into five tables and 
# 3. writes these tables in parquet format locally and to S3
# 
# Parameters for S3 are read from a credentials file
#  -- AWS Access Keys, region, S3 bucket names, local directory locations
#
# If the output bucket in S3 does not exist, it is created
# To speed up reading from S3 when creating a spark session an algorithm is used :
# -- mapreduce.fileoutputcommitter.algorithm.version", "2"
# Other libraries were used in spark.config for efficient processing
##########################################################################################################################
import configparser
from datetime import datetime
import os
import sys
import boto3
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int

# Setup config file with credentials file - dl.cfg
config = configparser.ConfigParser()
config.read('dl.cfg')

# Environment variables loaded with AWS access and secret keys 
os.environ['AWS_ACCESS_KEY_ID']=config.get('AWS','AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('AWS', 'AWS_SECRET_ACCESS_KEY')
os.environ['AWS_REGION']=config.get('AWS', 'REGION')

# S3 parameters for input and output buckets
# Note : input and output buckets will be in the same region

s3 = boto3.resource('s3',
                       region_name=os.environ['AWS_REGION'],
                       aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID'],
                       aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
                   )

def create_spark_session():
    """
      1. Creates a Spark Session with config info such as spark.jars.packages and
         hadoop-aws:2.7.0
      2. Uses parquet libraries to speed up writing to parquet files
      3. Uses a mapreduce algorithm to speed up the reading of udacity input file
      returns: a spark session
    """
    spark = SparkSession \
            .builder \
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
            .config("spark.hadoop.fs.s3a.multiobjectdelete.enable","false") \
            .config("spark.hadoop.fs.s3a.fast.upload","true") \
            .config("spark.sql.parquet.filterPushdown", "true") \
            .config("spark.sql.parquet.mergeSchema", "false") \
            .config("spark.sql.parquet.compression.codec", "gzip")\
            .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2") \
            .config("spark.speculation", "false") \
            .getOrCreate()              
    return spark

def check_bucket_exists (curr_region, bucket_name):
    """
      Create a bucket in S3 for output files if it is non-existent
      Arguments:
        curr_region : string
        bucket_name : string
    """
    try:
        s3.create_bucket(Bucket=bucket_name, 
                        CreateBucketConfiguration={'LocationConstraint': curr_region})
        print('Bucket Created : ' + bucket_name + ' : Creation Date : ' + str(s3.Bucket(bucket_name).creation_date))
    except:
        print('Bucket : ' + bucket_name + ' : Exists')
        
def process_song_data(spark, input_data, output_data, location):
    """      
      This function :
      1. reads song_data in json format (stored in an S3 bucket)
      2. extracts columns for the song table getting rid of duplicates
      3. writes songs in parquet format to an output S3 bucket
      4. extracts columns from songs dataframe to create artists table, getting rid of duplicates
      5. writes artist in parquet format
      
      Arguments:
      a. spark : Current spark session
      b. input_data : Type=string  : Local data or S3 bucket data
      c. output_data : Type=string : Parquet files to Local directory OR S3 bucket
      d. location : Type=string : Local Directory or S3
    """
    print ('===========================================================================================')
        
    SongDataSchema = R([
        Fld("artist_id", Str()),
        Fld("artist_latitude", Dbl()),
        Fld("artist_location", Str()),
        Fld("artist_longitude", Dbl()),
        Fld("artist_name", Str()),
        Fld("duration", Dbl()),
        Fld("num_songs", Int()),
        Fld("song_id", Str()),
        Fld("title", Str()),
        Fld("year", Int())
    ])
    
    # get filepath to song data file
    song_data = os.path.join(input_data + 'song_data/*/*/*/*.json')
    
    # read song data file
    print ('Start Time : Read song_data : ' + location + ' : ' + str(datetime.now()))
    
    song_data_df = spark.read.json(song_data, schema=SongDataSchema)
    
    print ('End Time : Read song_data : ' + location + ' : ' + str(datetime.now()))
    print ('===========================================================================================')
    
    # extract columns to create songs table
    songs_table = song_data_df.select(['song_id', 'title', 'artist_id', 'year', 'duration']).dropDuplicates()
    
    #songs_table.printSchema()
    
    # write songs table to parquet files partitioned by year and artist
    path_for_songs = os.path.join(output_data, 'songs')
    
    ################################# Write songs to parquet format ######################################################
    print ('Start Time : Writing songs : ' + location  + ' : ' + str(datetime.now()))
    
    songs_table.write \
               .option("compression", "gzip") \
               .mode('overwrite') \
               .partitionBy('year', 'artist_id') \
               .parquet(path_for_songs)
    
    print ('End Time : Writing songs to : ' + location + ' : ' + str(datetime.now()))
    print ('===========================================================================================')

    # ----------------------------- Artists  Table ------------------------------------------------
    # extract columns to create artists table
    artists_table = song_data_df.select(['artist_id', 'artist_name', 'artist_location',
                                         'artist_latitude', 'artist_longitude']).dropDuplicates()
            
    # rename fields according to dimension tables column names in Project Instructions
    artists_table = artists_table.withColumnRenamed('artist_name','name') \
                                 .withColumnRenamed('artist_location','location') \
                                 .withColumnRenamed('artist_latitude','latitude') \
                                 .withColumnRenamed('artist_longitude','longitude')
    
    #artists_table.printSchema()
    #artists_table.limit(3).show()
    
    # write artists table to parquet files
    path_for_artists = os.path.join(output_data, 'artists')
    
    ############################ Write artists to parquet format ############################################################
    
    print ('Start Time : Writing artists to : ' + location + ' : ' + str(datetime.now()))  
    
    artists_table.write \
                .option("compression", "gzip") \
                .mode("overwrite") \
                .parquet(path_for_artists)
    
    print ('End Time : Writing artists to : ' + location + ' : ' + str(datetime.now()))
    print ('===========================================================================================')


def process_log_data(spark, input_data, output_data, location):
    """
       1. Reads log_data in json format from AWS S3
       2. Extracts columns for users, time and songplays 
       3. Writes tables to AWS S3 in parquet format
       
       Arguments:
       a. spark : Current Spark Session
       b. input_data : Type=string : Local Data or S3 bucket data
       c. output_data : Type=string : Parquet files to Local Directory or S3 bucket
       d. location : Type=string : Local Directory or S3 bucket
    """
    
    # get filepath to log data file from s3 or Local diretory
    if location == "Local Directory":
        log_data = os.path.join(input_data, 'log_data/*.json')
    else:
        log_data = os.path.join(input_data, 'log_data/*/*/*.json')
    
    # read log data file
    print ('Start Time : Reading log_data from : ' + location + ' : ' + str(datetime.now()))
    
    log_data_df = spark.read.json(log_data)
    
    #log_data_df.printSchema()
    #log_data_df.limit(3).show()
    
    print ('End Time : Reading log_data from : ' + location + ' : ' + str(datetime.now()))
    print ('===========================================================================================')
    
    # filter by actions for song plays
    log_data_df = log_data_df.filter(log_data_df.page == 'NextSong')

    # extract columns for users table    
    users_table = log_data_df['userId', 'firstName', 'lastName', 'gender', 'level'].dropDuplicates()
    
     # rename fields according to dimension tables column names in Project Instructions
    users_table = users_table.withColumnRenamed('userId','user_id') \
                                 .withColumnRenamed('firstName','first_name') \
                                 .withColumnRenamed('lastName','last_name')                     
    
    #users_table.limit(3).show()
    
    ###################### Write users table to parquet format ##########################
    # write users table to parquet files   
    path_for_users = os.path.join(output_data, 'users')
    
    print ('Start Time : Writing users to : ' + location + ' : ' + str(datetime.now()))
    
    #users_table.printSchema()
    
    users_table.write \
               .option("compression", "gzip") \
               .mode("overwrite") \
               .parquet(path_for_users)
    
    print ('End Time : Writing users to : ' + location + ' : ' + str(datetime.now()))
    print ('===========================================================================================')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda ts:datetime.fromtimestamp(int(ts)/1000),T.TimestampType())
    # print ('Change ts to timestamp')    
    
    # create datetime column from original timestamp column   
    log_data_df = log_data_df.withColumn('start_time', get_timestamp('ts')) 
    
    # Get datetime columns from timestamp
    time_table = log_data_df.select('ts','start_time') \
                   .withColumn('hour',    F.hour('start_time')) \
                   .withColumn('day',     F.dayofmonth('start_time')) \
                   .withColumn('week',    F.weekofyear('start_time')) \
                   .withColumn('month',   F.month('start_time')) \
                   .withColumn('year',    F.year('start_time')) \
                   .withColumn('weekday', F.dayofweek('start_time')) \
                   .select('start_time', 'hour', 'day', 'week', 'month', 'year', 'weekday')
    
    #time_table.printSchema()
    
    # get time_table path       
    path_for_time_table = os.path.join(output_data, 'time')
    
    ################## Write time table to  parquet format ###########################
    # write time table to parquet files partitioned by year and month
    print ('Start Time : Write Time Table to : ' + location + ' : ' + str(datetime.now()))
    
    # write time_table to parquet file that is partitioned by year and month        
    time_table.write \
              .option("compression", "gzip") \
              .mode("overwrite") \
              .partitionBy("year", "month") \
              .parquet(path_for_time_table)
    
    print ('End Time : Write Time Table to : ' + location + ' : '  + str(datetime.now()))
    print ('===========================================================================================')

    # get the path for songs
    path_for_songs = os.path.join (output_data, 'songs')
    
    # get filepath to song data file
    song_data = os.path.join(input_data + 'song_data/*/*/*/*.json')
    
    SongDataSchema = R([
        Fld("artist_id", Str()),
        Fld("artist_latitude", Dbl()),
        Fld("artist_location", Str()),
        Fld("artist_longitude", Dbl()),
        Fld("artist_name", Str()),
        Fld("duration", Dbl()),
        Fld("num_songs", Int()),
        Fld("song_id", Str()),
        Fld("title", Str()),
        Fld("year", Int())
    ])
    
    # read song data file for songplays table
    print ('Start Time : Read song_data : ' + location + ' : ' + str(datetime.now()))
    
    song_df = spark.read.json(song_data, schema=SongDataSchema)
    
    print ('End Time : Read song_data : ' + location + ' : ' + str(datetime.now()))
    print ('===========================================================================================')
    
    # extract columns from joined song and log datasets to create songplays table 
    joined_song_log_df = log_data_df['ts', 'start_time', 'userId', 'level', 'song', 'artist', 'sessionId', 'location', 'userAgent'].dropDuplicates()
    joined_song_log_df.limit(5)
    
    log_and_song_df = joined_song_log_df.join(song_df, joined_song_log_df.song == song_df.title)
    
    ################ Create Songplays Table #####################################################
    
    print ('Start Time : Create Songplays table : ' + str(datetime.now()))
    
    songplays_table = log_and_song_df.select(
                      F.monotonically_increasing_id().alias('songplay_id'),
                      'start_time',
                      F.col('userId').alias('user_id'),
                      'level',
                      'song_id',
                      'artist_id',
                      F.col('sessionId').alias('session_id'),
                      'location',
                      F.year('start_time').alias('year'),
                      F.month('start_time').alias('month'),                     
                      F.col('userAgent').alias('user_agent')).dropDuplicates()  
    
    #songplays_table.printSchema()
    
    # get path for songplays    
    path_for_songplays = os.path.join(output_data, 'songplays')
    
    ############################### Write Songplays to Parquet format #################################
    
    # write songplays table to parquet files partitioned by year and month
    print ('Start Time : Write songplays to : ' + location + ' : ' + str(datetime.now()))
    
    # songplays_table.write.parquet(path_for_songplays, mode='overwrite', partitionBy=['year', 'month'])
    songplays_table.write \
                   .option("compression", "gzip") \
                   .mode("overwrite") \
                   .partitionBy(['year', 'month']) \
                   .parquet(path_for_songplays)
    
    print ('End Time : Write songplays to : ' + location + ' : ' + str(datetime.now())) 
    print ('===========================================================================================')
 

def input_output_location (spark, location):
    """
       1. This will process local data and save files to the Local Directory OR
       process S3 input data and save output to a new S3 bucket
       
       2. If Local Directory is selected:
          a. Read from local directory (a subset of files) - song_data and log_data in Udacity workspace
          b. Transform into tables and write tables in parquet format and save to local directory
          
       3. IF S3 is selected :
          a. Create an output bucket in S3 if it does not exist
          b. Read the full song_data and log_data from S3 bucket in us-west-2,
          c. Transform into tables and write to parquet format, saving it in a new S3 bucket in us-west-2
          
       arguments:
         spark : current spark session
         location : Local Directory or S3
    """
    
    if (location=='Local Directory'): 
        input_data  = config.get('LOCAL_DATA', 'INPUT')
        output_data = config.get('LOCAL_DATA', 'OUTPUT')
        title = 'Local Data and Directory'
    else:
        input_bucket_name  = config.get('S3', 'INPUT_BUCKET')
        output_bucket_name = config.get('S3', 'OUTPUT_BUCKET')   
        region = config.get('AWS', 'REGION')
        
        # if output S3 bucket does not exist, create a new bucket
        check_bucket_exists(region, output_bucket_name)
        input_data  = 's3a://' + input_bucket_name + '/'
        output_data = 's3a://' + output_bucket_name + '/'
        title = 'S3 Bucket'
    
    print ('****************************************************************************************************')
    print ('\ninput_data : ' + input_data)
    print ('\noutput_data : ' + output_data) 
    print ('\nlocation : ' + location)
    
    print ('****************************** ETL Processing Started - ' + title + ' *******************************')
    process_song_data(spark, input_data, output_data, location)    
    process_log_data(spark, input_data, output_data, location)
    print ('****************************** ETL Processing Completed - ' + title + ' *****************************')
      
def main():
    """
       1. Create a Spark Session
       2. Call the input_output function which processes a subset of files on the local directory OR
          S3 Bucket depends on what the user chooses when this script is run.    
    """
    spark = create_spark_session()
    
    # user will choose processing of local data or data on S3 and storage of parquet to
    # local output directory or S3 bucket
    
    processing = True
    n1=0
    while (processing):
        print ('****************************************************************************************************')
        print ('\nEnter 1 to process File subset in Local Directory' + 
               '\nEnter 2 to process full Input file in S3 Bucket and save parquet files to new S3 bucket' + 
               '\nEnter 3 to Quit')
        
        n1 = int(input('\nEnter a number: '))
        if (n1 == 1):
            location = 'Local Directory'
            break
        elif (n1 == 2):
            location = 'S3 Bucket'
            break
        elif (n1 == 3):
            sys.exit("\nExiting the Script : You entered : " + str(n1))
        else: 
            print ('\nError : You entered : ' + str(n1) + ' : Try Again' )
     
    # start processing - Local directory or S3
    input_output_location (spark, location)    
   
if __name__ == "__main__":
    main()
