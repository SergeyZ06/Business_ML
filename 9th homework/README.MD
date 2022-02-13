**9th_homework**

Deployment steps:
1. git clone https://github.com/SergeyZ06/Business_ML
2. cd Business_ML/
3. git checkout 9th_homework
4. cd 9th_homework/
5. sudo docker build -t sergeyz06/9th_homework .
6. sudo docker run -d -p 8180:8180 sergeyz06/9th_homework
7. python3 test_request.py

***

Service accessibility:

    http://0.0.0.0:8180//predict

Look file 'test_response.py' for more details.
