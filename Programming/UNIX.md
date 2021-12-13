# UNIX and Linux

## Commands

You can use `man <command>` to view the document of a command. Also see the cheatsheet [here](https://www.guru99.com/linux-commands-cheat-sheet.html).

Besides, you can clean the screen by `Ctrl + L`.

### File system

| Command | Function                                   |
| ------- | ------------------------------------------ |
| ls      | list files and folders (see also `tree`)   |
| cd      | change current directory                   |
| pwd     | show current directory                     |
| mkdir   | create directory                           |
| rm      | delete files or folders (see also `rmdir`) |
| mv      | move or rename file                        |
| cp      | copy                                       |
| du      | size of files and folders                  |

### I/O

| Command | Function                                                     |
| ------- | ------------------------------------------------------------ |
| cat     | show file content / create new file / concatenate files      |
| more    | show file content page by page (also see `less`)             |
| head    | show the head of a file (also see `tail`)                    |
| echo    | output a message to STDOUT; can used together with `>>` to write into files |

### Search

| Command | Function                                                     |
| ------- | ------------------------------------------------------------ |
| locate  | Search according to a maintained database; fast but not necessarily exhausted |
| find    | Search and process                                           |
| grep    | Global Regular Expression Print (to STDOUT)                  |

### Process Management

| Command | Function                                           |
| ------- | -------------------------------------------------- |
| ps      | View processes' status (see also `jobs`)           |
| fg      | Bring suspended jobs to foreground (see also `bg`) |
| kill    | Kill specific process                              |

From FSL:

- To kill a job that is running (in the "foreground") in the terminal just type `Ctrl-c`
- To get a list of jobs running the background, type `jobs`
- To bring something from running in the background to running in the foreground, type `fg` if it is the last job that was started.
- To force a job that is already running (in the foreground) to be in the background, type `Ctrl-z` and then `bg`. Note that `Ctrl-z` on its own will give you the terminal prompt back but the job will be sleeping, not running in the background, until you do `bg`.

### Other Commands

- `tar`: archive management
  - Modes (only 1 of the following can be used!)
    - `-c`: packing
    - `-t`: list the content of a tarfile
    - `-x`: unpacking
  - `-C <path>`: output to certain path
  - `-f <filename>`: specify filename (no more parameter is allowed between `-f` and `<filename>`!)
  - `-z`: compress (or uncompresse) with gzip (for `.tar.gz` files)

- Operators to run two command in one line:

  - `;` or `&&` (logical AND) or `||` (logical OR)

  - If you just want to execute two commands at once, `&&` is recommended since the second command will only be executed if the first one successfully returns. For example, you may cause disaster if you want to remove the content of a directory but type it wrong:

    ```bash
    cd <Typo> ; rm -rf *
    ```

    which will delete everything in your system.
    
  - Also see https://www.tecmint.com/chaining-operators-in-linux-with-practical-examples/

- Line break: use back slash `\` to continue your command on a new line

## Filesystem Hierarchy Standard (FHS)

### Folders in root Directory `/`

| Folder                             | Functions / Contents                                         |
| ---------------------------------- | ------------------------------------------------------------ |
| bin                                | System command binaries, e.g. `ls`                           |
| boot                               | Static files of boot loaders, e.g. kernal files              |
| dev                                | Device files                                                 |
| etc                                | Static configuration files                                   |
| home                               | User home directories, containing folders named after each user |
| lib (and optionally lib32 / lib64) | Shared libraries                                             |
| media                              | Mount (挂载) point for removable media (e.g. CD)             |
| mnt                                | Mount point for removable filesystems (e.g. U-disk)          |
| opt                                | Add-on application packages                                  |
| root                               | Home directory of the root user                              |
| run                                | Run-time system information                                  |
| sbin                               | System binaries                                              |
| srv                                | Services data                                                |
| tmp                                | Temporary files                                              |
| usr                                | Unix software resources (read-only, sharable between OS)     |
| var                                | Variable data files                                          |

Other directories under root:

| Folder         | Functions / Contents                                         |
| -------------- | ------------------------------------------------------------ |
| lost+found     | File fragmentations due to crash or unexpected shutdown, usually empty |
| proc (and sys) | Mapping of some system information in the memory (not hard drive!) |

Notes:

- When you log in as a user (e.g. `chen`), your home directory (here `/home/chen/`) can be shortened as `~`
- Softwares are usually installed in `/opt`  or `/usr/local`, exactly which can be tricky:
  - `/usr/local` contains folders like `/bin`, `/sbin`, thus suitable for softwares not managed by the system packager, but still following the standard unix deployment rules (e.g. the ones your build with `make`)
  - softwares installed in `/opt` will have their own directory and may not follow these rules
- See [here](https://refspecs.linuxfoundation.org/FHS_3.0/fhs/ch04.html) for more about the `/usr` directory

## Terminal, Console, TTY, Shell?

- History:
  - In the very early UNIX computers, the machine is huge and usually only the administrator will control it directly through some hardware (a console)
  - Since UNIX is the first multi-user computer structure, every user is allowed to interact with the machine (terminal) using a device to send command and receive output
  - This machine is often a teletypewriter (TTY)
  - In modern system, TTY, terminal and console are the same, which is an interface to send and receive texts
- Terminal emulator:
  - The word "terminal" in modern use actually means "terminal emulator", which is a program that simulates the behavior of traditional (physical) terminals and interacts with the shell
  - Terminal emulator usually uses command-line-interface (CLI), but the program itself usually runs in a graphical-user-interface (GUI) system (e.g. Windows)
  - Terminal emulator can provide functions like changing colors and font, command history, auto complete, copy and paste, etc.
  - Some examples:
    - Linux: Gnome-terminal, Konsole
    - Mac: iTerm2
    - Windows: Win32 Console, Windows Terminal
- Shell:
  - A shell is the outmost layer around the operating system, which exposes the system's services to the user
  - Shell interprets users' inputs, calls kernal APIs (and sometimes other applications) and handles the outputs. For example, when you input the command `cat foo.txt` in a shell, it will call `cat.exe` and the latter will call kernal APIs like `open`
  - Shell is just a special application. It calls system APIs just like other applications, and thus replacable by similar applications (e.g., Windows Explorer can be replaced by `cmd.exe`)
  - Shells can use terminals to accept user inputs and provide outputs, e.g. PowerShell use Win32 console as its interface
  - Examples:
    - UNIX: sh
    - Linux: bash
    - Windows: command prompt (`cmd.exe`), PowerShell

## Windows Subsystem for Linux (WSL)

- Essentially a virtual machine running Linux

- Can be installed by typing this command in Powershell (administrators):

  ```bash
  wsl --install -d <Distribution Name>
  ```

  Replace `<Distribution Name>` with the Linux distribution you want, e.g. `Ubuntu-18.04` (as used below). If you dismiss the `-d` parameter, it will be the latest version of Ubuntu.

  See [Docs from Microsoft](https://docs.microsoft.com/en-us/windows/wsl/install) for details.

- [Windows Terminal](https://docs.microsoft.com/en-us/windows/terminal/get-started) allows you handle multiple CLI programs (in WSL and Windows) at the same time.

- File systems:

  - In WSL, the windows file system will be regarded as mounted drives (like an U-disk), thus will be put in `/mnt/` directory (e.g. `/mnt/c/Windows/`)
  - In Windows, the WSL file system will be regarded as a network drive, and can be viewed by inputing the url in Windows file explorer: `\\wsl$\<Your Linux Version>\`
  - Therefore, it will be faster to store your projects in the file system that you will be working on, rather than accessing them through the method above

- Using GUI:

  - You can use `VcXsrv` on Windows to display WSL GUI applications.
  
  - Follow the instructions of FSL:
  
    - Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/files/latest/download), which will install a launcher (to `vcxsrv.exe`) called `XLaunch`
  
    - Open your Ubuntu Shell and enter these commands:
  
      ```bash
      echo "export DISPLAY=\$(grep nameserver /etc/resolv.conf  | awk '{print \$2; exit}'):0" >> ~/.bashrc
      echo "export LIBGL_ALWAYS_INDIRECT=1" >> ~/.bashrc
      ```
  
      which add some settings into you `~/.bashrc` file so that VcXsrv can work.
  
    - To use GUI in WSL, open XLaunch and select the options you want (I usually use `multiple windows` for display), but make sure to:
  
      - **Disselect** `Native OpenGL`
      - **Select** `Disable access control`
  
      in the `Extra settings` panel.
  
    - Besides, you make need to trust VcXsrv in your firewall.
  
  - You can save the configurations and use command line option `-run <Config>` to skip the setting and start the server silently. More exactly:
  
    - Save it to where VcXsrv is installed (so you don't need to specify the path in `<Config>`), usually `%ProgramFiles%\VcXSrv\` (i.e., `C:\Program Files\VcXsrv`)
    - Find the shortcut for XLaunch (or create one), go to Properties-Target and add `-run <Config>` after `"C:\Program Files\VcXsrv\xlaunch.exe"`
  
  - Besides, if you change the zooming ratio in Windows, the GUI display may be blurred. In this case, you need to open the properties of the shortcut, and select "override high DPI" as "Application" in the compatibility mode.
  
- Bugs:

  - You may find the following error when you launch the terminal:

    ```bash
    mkdir: cannot create directory '/home/<User Name>/.cache': Permission denied
    -bash: /home/<User name>/.cache/wslu/integration.sh: Permission denied
    ```

    - If you check the contents of your home directory with `ls -la ~`, you will find that `.cache` (and maybe also `.conda`) is owned by root, rather than you
    - Therefore, the Ubuntu for WSL integration script cannot write into this directory
    - It seems that this problem is caused by `sudo apt update` and `sudo apt upgrade`
    - Solution: delete this directory (anyway it's just cache) by `sudo rm -r ~/.cache`
    
  - VPN:

    - using Cisco Annyconnect for Linux in WSL2 will cause problem in DNS resolve
    - solution:
      - use VPN on Windows instead
      - you must connect to the VPN **after** lauching WSL!

## Cluster Computing

- Basics:
  - The high-performance computing (HPC) clusters are usually Linux servers equipped with some computing job managment softwares
    - WUSM CHPC uses Slurm on Centos 8
    - WUSTL RIS uses LSF
  - End-users usually use SSH to login to the server remotely. They:
    - have their own `$HOME` directory (`/home/username`) as the main workspace, and sometimes another directory for temporary usage (`/scratch/username` for CHPC)
    - have limited access to system directories:
      - e.g., they can run the commands in `/usr/bin`
      - but they cannot use `sudo` or write into system directories
  - Users submit bash scripts and specify the dependencies to the management system to perform the computation
    - Due to the possible conflict between dependencies for different jobs, the server usually manages an environment for each job
    - One traditional way (as it is in CHPC) is to prepare all possible versions of a dependency in the server, and use `module` to specify the environment variables for each job
    - Another way (as it is in RIS) is to use `docker` to encapsule all dependencies of a job into a standalone, platform-independent *container*
- `module`:
  - Use `module avail <package>` to see the available versions of a package
  - Use `module spider <RegExp>` to search for all available packages
  - Use `module load <package/version>` to load a specific version of a package into your environment
  - Use `module unload <package/version>` or `module del <package/version>` to unload

- `ssh` and `scp`:
  - Usually you need to start the ssh service by `sudo service ssh start`
  - Login to remote server: `ssh [parameters] [user@]host[:[port]] [command]`
    - frequently-used parameters:
      - `-p` (lowercase): specify the port to connected to (can also use `[user@]host:[port]` instead), by default port 22
      - `-Y`: enable X11 forwarding. You can run GUI on the server through X11 with this option.
    - `[command]` is executed on the remote host rather than local terminal
  - Logout: `logout` or `exit`
  - File transmission: `scp [parameters]  [[user@]host1:]file1 ... [[user@]host2:]file2`
    - `scp` use SSH protocol for security
    - frequently-used parameters:
      - `-P` (uppercase): specify the port to connected to (by default also port 22)
      - `-r`: recursively copy the whole folder and its contents
    - Examples:
      - Upload a local file to a remote directory: `scp <local_file> <user>@<host>:<directory>`
      - Same as above but change the name of the uploaded file: change `<directory>` to `<directory>/<new_filename>`
      - Upload a local folder to a remote directory: `scp -r <local_folder> <user>@<host>:<directory>`
      - Download is similar, just switching the last two parameters
