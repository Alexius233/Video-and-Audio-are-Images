import os
import cv2
import shutil


root = "/root/autodl-tmp/VMR_Data"
newroot = "/root/autodl-tmp/VMR_PRO"

labels = os.listdir(root)
#print(labels)



for label in sorted(labels):
    dir = os.path.join(newroot, label)  # 查看新位置是否有文件夹，没有就新建（valid,test,train三种）
    if not os.path.exists(dir):
        os.mkdir(dir)

    root_dir = os.path.join(root, label)  # 老位置具体视频的位置指示（valid,test,train三种）

    for types in sorted(os.listdir(root_dir)): # 进入了老位置下一级并遍历（audio,video两种)

        if types == 'audio':
            root_audio = os.path.join(root_dir, types) # 老地址的audio
            audio_dir = os.path.join(dir, types) # 建立新地址的audio
            if not os.path.exists(audio_dir):
                os.mkdir(audio_dir)

            for count in sorted(os.listdir(root_audio)):   # 在audio里遍历i.wav
                shutil.copyfile(os.path.join(root_audio, count), os.path.join(audio_dir, count)) # 复制过去
                fa = open(os.path.join(dir, "audiofilename.txt"), "a")   # 建立txt
                fa.write(os.path.join(audio_dir, count))    # 写上新地址
                fa.write('\n')
                fa.close()

        if types == 'video':
            root_video = os.path.join(root_dir, types)   # 老地址video
            video_dir = os.path.join(dir, types)  # 建立新地址的video
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)

            for count in sorted(os.listdir(root_video)):  # 在video内部遍历i.mp4
                next_root_video = os.path.join(os.path.join(root_dir, types), count)  # 到了老video/i.mp4
                if not os.path.exists(os.path.join(video_dir, count)):   # 建立新video/i.mp4
                    os.mkdir(os.path.join(video_dir, count))

                print(next_root_video)
                vc = cv2.VideoCapture(next_root_video)  # 读入视频

                fps = round(vc.get(int(cv2.CAP_PROP_FPS)))  # 读取码率，每秒几帧
                size = int(vc.get(int(cv2.CAP_PROP_FRAME_COUNT)))  # 长度，总帧数
                gap = int(size / 9)          # 设置为抽9帧

                fv = open(os.path.join(dir, "videofilename.txt"), "a")   # 建立txt
                fv.write(os.path.join(video_dir, count))    # 写上新地址
                fv.write('|')
                fv.write(str(round(fps)))   # 写fps
                fv.write('\n')
                fv.close()

                c = - int(0.5 * fps) + gap # 开始设置在-0.5fps，即过视频有偏移量防止最后一帧实在最有一张图(这常常没有有意义信息)
                number = 0
                num = 0
                rval = vc.isOpened()
                print(rval)

                while rval:  # 循环读取视频帧

                    number += 1

                    rval, frame = vc.read()

                    if rval :
                        if number == c :
                            num += 1
                            pic_root = os.path.join(os.path.join(video_dir, count), str(num) + '.png')
                            cv2.imwrite(pic_root, frame)
                            #print("已经写入")
                            cv2.waitKey(1)
                            c = c + gap
                            if num == 9:
                                break

                    else:
                        break

                vc.release()

