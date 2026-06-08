# regex_spec.py：定义日志关键词扫描器使用的字符串正则规则
REGEX_SPECS = {
    "DATE": "[0-9]{4}-[0-9]{2}-[0-9]{2}",
    "TIME": "[0-9]{2}:[0-9]{2}:[0-9]{2}",
    "LEVEL": "(INFO|WARN|ERROR|DEBUG|TRACE)",
    "IP": "[0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}",
    "STATUS": "[0-9]{3}",
    "USER": "[A-Za-z_][A-Za-z0-9_]*",
}

PRIORITY = ["IP", "DATE", "TIME", "LEVEL", "STATUS", "USER"]

