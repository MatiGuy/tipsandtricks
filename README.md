#### checkout already branch
```
git checkout XXXX 
```

#### create new brach
```
git checkout -b xxxx
```

#### first push to remote (only first time)
```
git push -u origin xxxx
```

#### create branch from remote 
```
git checkout -b XXXX origin/XXX
```

#### back to latest committed
```
git reset --hard HEAD
git reset COMMIT_ID //git log
```

#### show markdown tab
```
ctrl+shift+v
```

#### remove the file from the Git repository and the filesystem
```
git rm XXX
git commit -m "message"
git push origin branch_name  
```

#### remove the file only from the Git repository and not remove it from the filesystem
```
git rm --cached XXXX
git commit -m "message"
git push origin branch_name  

```
#### Extensions
- Commuinity Material Theme
- GitLens
- Markdown All in One
- Material Icon Theme
- Material Theme



