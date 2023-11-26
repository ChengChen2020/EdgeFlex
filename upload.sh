#!/usr/bin/expect

set pass "12345\n"
set file "840968.tar.gz"
set devices [list "cheng@128.46.74.158:/home/cheng" "dcsl@128.46.74.214:/home/dcsl" "xiang@128.46.74.171:/home/xiang"]
# set devices [list "cheng@128.46.74.158:/home/cheng"]
# set devices [list "dcsl@128.46.74.214:/home/dcsl"]
# set devices [list "xiang@128.46.74.171:/home/xiang"]

foreach d $devices {
	spawn scp /Users/chen4384/Desktop/EdgeFlex/checkpoint/$file $d
	expect "password:"
	send "$pass"
	interact
}