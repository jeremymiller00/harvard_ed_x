/* 
All data resides in one large table

columns:
course_id,userid_DI,registered,viewed,explored,certified,final_cc_cname_DI,LoE_DI,YoB,gender,grade,start_time_DI,last_event_DI,nevents,ndays_act,nplay_video,nchapters,nforum_posts,roles,incomplete_flag

To run this script, first create the database from the psql command line:

```
psql
CREATE DATABASE name
```

Then from the bash command line:

```
psql name < create_database.sql
```
*/

-- create student data table
CREATE TABLE harvard_data
(
    course_id VARCHAR(99),
    userid_DI VARCHAR(20),
    registered INT,
    viewed INT,
    explored INT,
    certified INT,
    final_cc_cname_DI VARCHAR(100),
    LoE_DI VARCHAR(20),
    YoB VARCHAR(20),
    gender VARCHAR(5),
    grade DECIMAL,
    start_time_DI DATE,
    last_event_DI DATE,
    nevents INT,
    ndays_act INT,
    nplay_video INT,
    nchapters DECIMAL,
    nforum_posts INT,
    roles INT,
    incomplete_flag INT,
    PRIMARY KEY(course_id, userid_DI)
);
