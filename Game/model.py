from multiprocessing import Process,Queue,Pipe
from game import send_data

if __name__ == '__main__':
    parent_conn,child_conn = Pipe()
    p = Process(target=send_data, args=(child_conn,))
    p.start()
    print(parent_conn.recv())
