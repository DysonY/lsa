# LSA
Perform LSA with 7 topics and 10 terms per topic:  
```bash
python topics.py 7 10
```

Write the output clusters to a file:
```bash
python topics.py 7 10 > output.txt
``` 

The topics.lsa() function returns a list of tuples containing the words for each topic.
