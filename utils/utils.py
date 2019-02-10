def print_time_info(start_time, end_time):
    print('[INFO] Start and end time of the last session: %s - %s'
          % (start_time.strftime('%d.%m.%Y %H:%M:%S'), end_time.strftime('%d.%m.%Y %H:%M:%S')))
    print('[INFO] Total time previous session took:', (end_time - start_time), '\n')
