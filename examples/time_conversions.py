# time_conversions.py
import datetime
import textwrap

import littletable as lt

process_data = textwrap.dedent("""\
    timestamp,eqpt,event,lot
    2020/01/01 04:15:22,DRILL01,LotStart,PCB146
    2020/01/01 04:16:02,DRILL01,Tool1,PCB146
    2020/01/01 04:19:47,DRILL01,Tool2,PCB146
    2020/01/01 04:26:03,DRILL01,LotEnd,PCB146
    """)

transforms = {'timestamp': lt.Table.parse_datetime("%Y/%m/%d %H:%M:%S")}

# get timestamp of first event
data = lt.Table().csv_import(process_data, transforms=transforms, limit=1)
start_time = data[0].timestamp

# read in values, converting timestamps to offset from start time
transforms = {'timestamp': lt.Table.parse_timedelta("%Y/%m/%d %H:%M:%S", reference_time=start_time)}
data = lt.Table(f"Events relative to {start_time}").csv_import(process_data, transforms=transforms)

data.present()


process_data = textwrap.dedent("""\
    elapsed_time,eqpt,event,lot
    0:00:00,DRILL01,LotStart,PCB146
    0:00:40,DRILL01,Tool1,PCB146
    0:03:45,DRILL01,Tool2,PCB146
    0:06:16,DRILL01,LotEnd,PCB146
    """)

transforms = {'elapsed_time': lt.Table.parse_timedelta("%H:%M:%S")}
data = lt.Table(f"Process step elapsed times").csv_import(process_data, transforms=transforms)

data.present()
