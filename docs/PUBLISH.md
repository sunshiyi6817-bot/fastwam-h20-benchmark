# 上传GitHub

先确认公司允许发布这些代码，建议在GitHub创建私有空仓库，不初始化README或LICENSE。将YOUR_USER和YOUR_REPO换成真实名称。

    git init -b main
    git add .
    git status --short
    git diff --cached --stat
    git commit -m "Add FastWAM H20 NGC 25.09 benchmark snapshot"
    git remote add origin git@github.com:YOUR_USER/YOUR_REPO.git
    git push -u origin main
    git rev-parse HEAD

最后一个命令输出的commit应随报告交付。此交付脚本没有替你创建提交、分支或推送。

SSH未配置时可使用HTTPS：

    git remote set-url origin https://github.com/YOUR_USER/YOUR_REPO.git
    git push -u origin main

使用GitHub批准的凭据管理器、GitHub CLI登录或PAT认证，不要把Token写进URL、脚本或仓库。若git commit提示身份缺失，仅为当前仓库配置自己的user.name和user.email，再重试。
