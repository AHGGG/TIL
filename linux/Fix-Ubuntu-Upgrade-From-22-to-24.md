# Ubunut22.04升级24.04崩溃修复
记录一下崩溃修复的过程, 下次好有个参考.  

我的电脑里有两块硬盘，第一块装的是Ubuntu 22, 一块装的是window系统。  


最近看到hyprland太好看了, 就想升级到Ubuntu 24，结果升级过程中系统直接崩了。重启后所有的recovery mode进去都是emergency状态, 遇到的几个问题:  
1. recovery mode里修复package, 但是报错: An upgrade from 'noble' to 'jammy' is not supported with this tool.  
2. 使用apt --fix-broken install, 但是报错: dpkg: error processing archive /var/cache/apt/archives/libpython3.12-minimal_3.12.3-1ubuntu0.8_amd64.db (--unpack)  
3. recovery mode进行enable network的时候报错: NetworkManager.service failed because the control process exited with error code. 日志里也看到不到什么有用的信息.  


之前的系统没有备份... 所以最后只有使用live usb启动, 然后尝试chroot到原来的系统里进行修复. 但是还是遇到上面的问题.  
无奈只有重装系统, 把之前系统里的东西mount出来, 这样就行了. 安装的时候看到了好几种不同的分区选项, 选择了LVM(说可以动态分配空间), 重装完之后的样子:
```text
/dev/sda（120G）：旧系统，已崩溃
    sda1: EFI分区（512M）
    sda2: Linux系统分区（118.7G）
/dev/sdb（465G）：新系统
    sdb1: EFI分区（1G）
    sdb2: /boot分区（2G）
    sdb3: LVM物理卷（462.7G，包含根分区）
```

在claude-sonnet的帮助下, 一步步解决了问题, 命令:
```bash
# 激活LVM. 注意, 因为我重装系统的时候选择的是LVM, 所以这里需要激活LVM
sudo vgscan
sudo vgchange -ay

# 查看LVM信息
sudo lvdisplay

# 挂载根分区（注意是LVM逻辑卷）
sudo mount /dev/mapper/ubuntu--vg-ubuntu--lv /mnt

# 挂载boot分区
sudo mount /dev/sdb2 /mnt/boot

# 挂载EFI分区
sudo mount /dev/sdb1 /mnt/boot/efi

# 挂载必要的系统目录（为了chroot）
sudo mount --bind /dev /mnt/dev
sudo mount --bind /proc /mnt/proc
sudo mount --bind /sys /mnt/sys
sudo mount --bind /run /mnt/run

# 挂载efivarfs, 因为后面重新安装GRUB的报错, 所以需要这一步
sudo mount -t efivarfs efivarfs /mnt/sys/firmware/efi/efivars

# 进入新系统
sudo chroot /mnt

# 重新安装GRUB
grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=ubuntu --recheck
update-grub

# 退出chroot
exit

# 检查EFI启动项
sudo efibootmgr -v

# 检查GRUB配置
sudo cat /mnt/boot/grub/grub.cfg | grep menuentry

# 检查fstab
sudo cat /mnt/etc/fstab

# 确认没问题后，卸载所有分区
sudo umount -R /mnt

# 重启
sudo reboot
```

然后因为BIOS还是从旧的硬盘启动的, 搞了半天, 各种命令, 但是最后直接从BIOS里选择启动盘就行了, 选择新的硬盘就可以进入新的系统了.

进入新系统后, 将原来就硬盘里的数据拷贝出来就行.  

然后装上timeshift, 进行备份. 事实证明, 备份还是很有必要的. 因为折腾hyprland, 又把系统弄崩了.... 前几个都是好的, 但是断电自动关机后, 再次启动就直接进入emergency模式了. 这次直接用timeshift恢复到了折腾hyprland之前的状态.....