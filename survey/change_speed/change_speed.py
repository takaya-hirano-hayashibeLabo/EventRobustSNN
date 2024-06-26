import tonic
from pathlib import Path
import numpy as np
from copy import deepcopy


def event2anim(events, time_window, max_x, max_y, output_path=Path(__file__).parent, file_name="output.mp4", scale_factor=10):
    """
    Convert event data to an animation and save as an mp4 file.

    Parameters:
    events (numpy structured array): The event data [(pixel_x, pixel_y, timestep, spike), ...]
    time_window (int): The time window for each frame.
    max_x (int): The maximum x value of the pixels.
    max_y (int): The maximum y value of the pixels.
    output_file (str): The path to the output mp4 file.
    scale_factor (int): Factor by which to scale the resolution.
    """
    import subprocess
    import os
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import viridis,plasma,cividis

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create a video writer object with increased resolution
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    scaled_max_x = max_x * scale_factor
    scaled_max_y = max_y * scale_factor
    tmpout = str(output_path / "tmp.avi")
    out = cv2.VideoWriter(tmpout, fourcc, 30.0, (scaled_max_x, scaled_max_y))

    # Determine the maximum timestep
    max_t = events[-1][2] + 1

    for frame_start in range(0, max_t, time_window):
        frame_end = frame_start + time_window

        frame = np.ones((scaled_max_y, scaled_max_x), dtype=np.uint8)*0.5

        # Add events to the frame
        for event in events:
            x, y, t, p = event
            if frame_start <= t < frame_end:
                scaled_x = x * scale_factor
                scaled_y = y * scale_factor
                color = 1 if p == 1 else 0  # Green for +1, Blue for -1
                for i in range(scale_factor):
                    for j in range(scale_factor):
                        frame[scaled_y + i, scaled_x + j] = color

        # Normalize the grayscale frame
        norm = Normalize(vmin=0, vmax=1)
        frame_normalized = norm(frame)

        # Apply the viridis colormap
        frame_colored = viridis(frame_normalized)

        # Convert the frame to uint8
        frame = (frame_colored * 255).astype(np.uint8)

        # Write the frame to the video
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Release the video writer object
    out.release()

    # Re-encode the video using ffmpeg
    file_name = file_name + ".mp4" if not ".mp4" in file_name else file_name
    ffmpeg_command = [
        'ffmpeg', '-y', '-i', tmpout,
        '-pix_fmt', 'yuv420p', '-vcodec', 'libx264',
        '-crf', '23', '-preset', 'medium', str(output_path / file_name)
    ]
    subprocess.run(ffmpeg_command)
    # Remove the temporary file
    os.remove(tmpout)



def event2anim_accumulating(events, time_window, max_x, max_y, output_path=Path(__file__).parent, file_name="output.mp4"):
    """
    Convert event data to an accumulating animation and save as an mp4 file.

    Parameters:
    events (numpy structured array): The event data [(pixel_x, pixel_y, timestep, spike), ...]
    time_window (int): The time window for each frame.
    max_x (int): The maximum x value of the pixels.
    max_y (int): The maximum y value of the pixels.
    output_file (str): The path to the output mp4 file.
    """
    import subprocess
    import os
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    tmpout = str(output_path / "tmp.avi")
    out = cv2.VideoWriter(tmpout, fourcc, 30.0, (max_x, max_y))

    # Determine the maximum timestep
    max_t = events[-1][2] + 1

    # Create a blank frame for accumulation with white background
    accumulated_frame = np.ones((max_y, max_x, 3), dtype=np.uint8) * 255

    # Process events in time windows
    for frame_start in range(0, max_t, time_window):
        frame_end = frame_start + time_window

        # Add events to the accumulated frame
        for event in events:
            x, y, t, p = event
            if frame_start <= t < frame_end:
                if p == 1 and np.array_equal(accumulated_frame[y, x], [255, 255, 255]):
                    accumulated_frame[y, x] = [255, 0, 0]  # Red for +1
                elif p == -1 and np.array_equal(accumulated_frame[y, x], [255, 255, 255]):
                    accumulated_frame[y, x] = [0, 0, 255]  # Blue for -1


        # Write the frame to the video
        out.write(cv2.cvtColor(accumulated_frame, cv2.COLOR_RGB2BGR))

    # Release the video writer object
    out.release()

    # Re-encode the video using ffmpeg
    file_name = file_name + ".mp4" if not ".mp4" in file_name else file_name
    ffmpeg_command = [
        'ffmpeg', '-y', '-i', tmpout,
        '-pix_fmt', 'yuv420p', '-vcodec', 'libx264',
        '-crf', '23', '-preset', 'medium', str(output_path / file_name)
    ]
    subprocess.run(ffmpeg_command)
    # Remove the temporary file
    os.remove(tmpout)


def change_speed(events, alpha):
    """
    :param events: [(pixel_x, pixel_y, timestep, spike), ...]
    :param alpha: 速度倍率. 2にすれば2倍の速度になる
    """
    new_events=deepcopy(events)
    for i in range(len(events)):
        new_events[i][2] = deepcopy(int(new_events[i][2] / alpha))
    return new_events


def change_speed_v2(event, params:list):
    """
    指定した速度とフレーム比でイベントスピードを変換する

    :param event: [(pixel_x, pixel_y, timestep, spike), ...]
    :param params: [{speed, rate},...]
    """


    #>> いい感じの比で速度変換の倍率リストを生成 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    split_rate=[]
    speeds=[]
    for p in params:
        speed=p["speed"]
        rate=p["rate"]
        speeds+=[speed]
        split_rate+=[speed*rate]

    event_num=len(event)
    speed_trj=np.zeros(event_num)
    idx=0
    for i, (s,r) in enumerate(zip(speeds,split_rate)):
        step=int(event_num*(r/np.sum(split_rate)))

        if i<len(speeds)-1:
            speed_trj[idx:idx+step]=s
            idx+=step
        else:
            speed_trj[idx:]=s
    #<< いい感じの比で速度変換の倍率リストを生成 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> 速度変換 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    dts=[]
    for i in range(1,len(event)):
        dts.append(int((event[i][2]-event[i-1][2])/speed_trj[i]))
    new_event=deepcopy(event)
    for i in range(1,len(event)):
        new_event[i][2]=new_event[0][2]+sum(dts[:i])
    #<< 速度変換 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    return new_event

def main():

    dataset = tonic.datasets.NMNIST(save_to=str(Path(__file__).parent.parent/"train_dvs/data"), train=True)
    datasize=tonic.datasets.NMNIST.sensor_size
    print(datasize)
    events, target = dataset[40000]
    print(events.shape)

    window_size=1400
    outpath=Path(__file__).parent/"videos"
    event2anim(
        events=events,
        time_window=window_size,
        max_x=datasize[0],
        max_y=datasize[1],
        file_name="original_speed",
        output_path=outpath
    )


    # alpha=0.5
    # new_events=change_speed(events, alpha)
    # event2anim(
    #     events=new_events,
    #     time_window=window_size,
    #     max_x=datasize[0],
    #     max_y=datasize[1],
    #     file_name=f"{alpha}times_speed",
    #     output_path=outpath

    # )

    # alpha=2.0
    # new_events=change_speed(events, alpha)
    # event2anim(
    #     events=new_events,
    #     time_window=window_size,
    #     max_x=datasize[0],
    #     max_y=datasize[1],
    #     file_name=f"{alpha}times_speed",
    #     output_path=outpath

    # )

    params=[
        {"speed":0.3, "rate":1.0},
        {"speed":2.0, "rate":0.5},
        {"speed":0.3, "rate":1.0},
    ]
    new_events=change_speed_v2(events, params)
    event2anim(
        events=new_events,
        time_window=window_size,
        max_x=datasize[0],
        max_y=datasize[1],
        file_name="changed_speed",
        output_path=outpath
    )

if __name__ == "__main__":
    main()