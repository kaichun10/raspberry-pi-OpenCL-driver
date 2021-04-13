# RaspberryPi-OpenCL-Driver

### Resize Raspberry Pi image and emulate with QEMU
[![IMAGE ALT TEXT HERE](https://raw.githubusercontent.com/kaichun10/RaspberryPi-OpenCL-Driver/README/img/6074df6944174-fbutube-Screenshot%20(124).png)](https://www.youtube.com/watch?v=5_DzkrMDxnc)

### Steps
1. Update and install QEMU
sudo apt-get update
sudo apt-get install qemu

2. Install the lastest Raspian Pi OS Image:
https://www.raspberrypi.org/software/operating-systems/

3. Get Raspberry-Pi linux kernel from repo:
https://github.com/dhruvvyas90/qemu-rpi-kernel

4. Rename extracted folder to:
raspbian_bootpart

5. Increase the Image size to allow for future upgrades

1) Check the disk-partition
fdisk 2021-01-11-raspios-buster-armhf-lite.img
p   print the partition table

2) Increase Image size by 6G
qemu-img resize 2021-01-11-raspios-buster-armhf-lite.img +6G

3) Modify the disk-partition
fdisk 2021-01-11-raspios-buster-armhf-lite.img

p   print the partition table

d   delete a partition

2

n   add a new partition

p   primary

2

532480  Starting address of 2nd partition

### ***IMPORTANT***

##### DO you want to remove the signature?

##### Enter N

w   write table to disk and exit

4) Check the result
fdisk 2021-01-11-raspios-buster-armhf-lite.img

6. Create a temp folder

7. Mount Image and check files
sudo mount -v -o offset=272629760 -t ext4 2021-01-11-raspios-buster-armhf-lite.img temp
cd temp
sudo vim ./etc/ld.so.preload
sudo vim ./etc/fstab

8. Unmount the file
cd ..
sudo umount temp


9. Start QEMU:
qemu-system-arm -kernel raspbian_bootpart/kernel-qemu-4.14.79-stretch -dtb raspbian_bootpart/versatile-pb.dtb -m 256 -M versatilepb -cpu arm1176 -serial stdio -append "rw console=ttyAMA0 root=/dev/sda2 rootfstype=ext4  loglevel=8 rootwait fsck.repair=yes memtest=1" -drive file=2021-01-11-raspios-buster-armhf-lite.img,format=raw -net nic -net user,hostfwd=tcp::5022-:22 -no-reboot


10. Check the resized image
df -h

11. Update Raspberry Pi
sudo apt update
sudo apt upgrade
sudo apt-get autoremove


### POSIX Threads:
[Useful-POSIX-Tutorials](https://computing.llnl.gov/tutorials/pthreads/)

### OpenCL-Specification for making OpenCL API calls:
[OpenCL-Specification.pdf](https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf)
