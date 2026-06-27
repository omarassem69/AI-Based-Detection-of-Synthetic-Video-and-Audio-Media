
processes = []

n = int(input("Enter number of processes: "))

for i in range(n):
    pid = int(input(f"\nEnter Process ID for process {i+1}: "))
    arrival = int(input("Enter Arrival Time: "))
    burst = int(input("Enter Burst Time: "))
    processes.append((pid, arrival, burst))


processes.sort(key=lambda x: x[1])

current_time = 0
total_waiting_time = 0

print("\nProcess Execution Order:")
for pid, arrival, burst in processes:
    start_time = max(current_time, arrival)
    waiting_time = start_time - arrival
    total_waiting_time += waiting_time
    current_time = start_time + burst
    
    print(f"Process {pid} -> Waiting Time = {waiting_time}")

print("\nTotal Waiting Time:", total_waiting_time)