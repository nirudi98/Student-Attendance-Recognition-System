# Student-Attendance-Recognition-System
Student Attendance System implemented using Python, can recognize whether a student is absent or present using image processing techniques. It can also compare several signing sheets and identify the similarity between the signatures

*** Please reload the project in pycharm after the code is successfully completed to see the newly written images. 


1. Project Folder name - StudentAttendanceSystem (this is inside code folder)

2. Folder Structure 
   CGVAssignment
     - signatures = student signatures of signing sheets from sheet One - Five are stored in separate folders
   
   input signing sheet images (1.jpeg/ 2.jpeg/ 3.jpeg/ 4.jpeg/ 5.jpeg)   

   other images produced during runtime will be stored in this folder
 
   Code_Images - student signatures cropped during runtime are stored here with their index number

   csv files of daily student attendance (already created attendance sheets are in the folder for your reference)

3. Main python files in the project
   - sams.py
   - Database.py
   - infovis.py
   - investigate.py

4. Please run the code using 1.jpeg/ 4.jpeg/ 5.jpeg one of these images. 

5. The cropped signatures will be saved in the given folder paths. 

6. Make sure to change the folder path according to your project file destination.

7. To run the Database.py file, uncomment the database creation code to create the database. 

8. When running the Database.py and infovis.py, you can change the csv file name accordingly to view student attendance on a different date.

9. Before running infovis.py, please install Jinja2 using 'pip install Jinja2'
