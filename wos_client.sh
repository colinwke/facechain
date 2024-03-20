#!/bin/bash
# bashsupport disable=HttpUrlsUsage,BP2001
# shellcheck disable=SC2166,SC2012,SC2086,SC2004,SC2308,SC2046

USAGE=$( # this is a block comment
    cat <<'EOF'
https://docs.58corp.com/#/space/1530209735373017088
EOF
)

version="wos-client(bash) 1.0.0"

tokenserver=""
wosserver=""
appkey=""
secret=""
bucket=""
filename=""
localfile=""
ttl=0
other=""

level=0 # debug=0 info=1 warning=2 failed=3

debug() {
    if [ $# -eq 1 -a $level -le 0 ]; then
        echo -e "[DEBUG]\t $1"
    fi
}

info() {
    if [ $# -eq 1 -a $level -le 1 ]; then
        echo -e "[INFO]\t $1"
    fi
}

warn() {
    if [ $# -eq 1 -a $level -le 2 ]; then
        echo -e "[WARN]\t $1"
    fi
}

panic() {
    if [ $# -eq 1 -a $level -le 3 ]; then
        echo -e "[PANIC]\t $1"
        exit 1
    fi
}

help() {
    echo "wos-client(bash) 1.0.0

usage: 
    ./wos-client.sh -a appid -b bucket -f filename -t tokenserver -h wosserver [-l file] [--precheck] [-ttl num_hour] upload 
    ./wos-client.sh -a appid -b bucket -f filename -t tokenserver -h wosserver delete
    ./wos-client.sh -a appid -b bucket -f filename -t tokenserver -h wosserver --ttl num_hour set-ttl
    "
    exit 0
}

command -v python3 >/dev/null 2>&1 || panic "must install python3"

getsize() {
    ls -l $1 | awk '{print $5}'
}

needprecheck=0

while true; do
    if [ $# -eq 0 ]; then
        break
    fi

    case "$1" in
    -v)
        echo $version
        exit 0
        ;;
    --help)
        help
        ;;
    -t)
        shift
        tokenserver="$1"
        ;;
    -h)
        shift
        wosserver="$1"
        ;;
    -a)
        shift
        appkey="$1"
        ;;
    -s)
        shift
        secret="$1"
        ;;
    -b)
        shift
        bucket="$1"
        ;;
    -f)
        shift
        filename="$1"
        ;;
    -l)
        shift
        localfile="$1"

        if [ ! -f $localfile ]; then
            panic "localfile $localfile not exist"
        fi
        ;;
    --ttl)
        shift
        ttl="$1"
        ;;
    --precheck)
        needprecheck=1
        ;;
    *)
        other="$other $1"
        ;;
    esac
    shift
done

if [ -z $localfile ]; then
    localfile=$filename
fi

userinfo() {
    debug "tokenserver: $tokenserver"
    debug "wosserver: $wosserver"
    debug "appkey: $appkey"
    debug "secret: $secret"
    debug "bucket: $bucket"
    debug "filename: $filename"
    debug "localfile: $localfile"
    debug "other: $other"
}

getjson() {
    echo "$2" | python3 -c "import sys, json; data=json.load(sys.stdin); print($1)"
}

gettoken() {
    resp=$(curl -s "http://$tokenserver/get_token?bucket=$bucket&filename=$filename&op=upload" -u $appkey:$secret)
    code=$(getjson "data['code']" "$resp")
    if [ $code -eq 0 ]; then
        getjson "data['data']" "$resp"
    else
        panic "get token error: $(getjson "data['message']" "$resp")"
    fi
}

uploadsmall() {
    token=$(gettoken)
    #echo -e ${token} > x.txt
    #token="ZUlSdjFMQXUyWC9mZDUwRllCWUFTS0RRa3pJPTpmPXRlc3Q2LmFtciZlPTE3MDYzNTY2NTEmcj02MzU3MDc1MzUmb3A9dXBsb2Fk"
    echo -e "${token}"
    #exit 1
    echo -e "${version}"
    echo -e "http://$wosserver/$appkey/$bucket/$filename"

    #token="TzJ4cTlCR2hQRFJSYVQ1SnFDQUczUHBuaUlFPTpmPW1vbml0b3JfdGVzdC5qcGcmZT0xNzA2MzY0Mzk4JnI9MzE3MzQ3MzY3MCZvcD11cGxvYWQ%3D"
    #filename="monitor_test.jpg"
    #bucket="vidorm"
    resp=$(curl -s --location --request POST "http://$wosserver/$appkey/$bucket/$filename" \
        --header "Authorization: $token" \
        --header "User-Agent: $version" \
        --header "Accept: */*" \
        --header "Connection: keep-alive" \
        --form "op=upload" \
        --form "ttl=$ttl" \
        --form "filecontent=@$localfile")

    echo -e ${resp}
    code=$(getjson "data['code']" "$resp")
    if [ $code -eq 0 ]; then
        # info "upload success,url: $(getjson data['data']['url'] $resp) intranet_url: $(getjson data['data']['url'] $resp) access_url: $(getjson data['data']['url'] $resp)"
        echo $resp
    else
        panic "get token error: $(getjson "data['message']" "$resp")"
    fi
}

uploadinit() {
    token=$(gettoken)

    resp=$(curl -s --location --request POST "http://$wosserver/$appkey/$bucket/$filename" \
        --header "Authorization: $token" \
        --header "User-Agent: $version" \
        --header "Accept: */*" \
        --header "Connection: keep-alive" \
        --form "op=upload_slice_init" \
        --form "ttl=$ttl" \
        --form "filesize=$size" \
        --form "slice_size=4194304")

    code=$(getjson "data['code']" "$resp")
    if [ $code -eq 0 ]; then
        session=$(getjson "data['data']['session']" "$resp")
    else
        panic "$resp"
    fi
}

needUpload=""

uploadslice() {
    offset=0
    for i in $(seq $slicenum); do
        file=$(printf "$localfile.tmp.%.8d\n" $(($i - 1)))

        if [ ! -f $file ]; then
            panic "couldn't find file $file"
        fi

        if [[ $needprecheck -eq 1 ]] && [[ $needUpload != *" $offset "* ]]; then
            debug "offset:$offset has uplaod"
            offset=$(($offset + 4194304))
            rm $file
            continue
        fi

        token=$(gettoken)

        resp=$(curl -s --location --request POST "http://$wosserver/$appkey/$bucket/$filename" \
            --header "Authorization: $token" \
            --header "User-Agent: $version" \
            --header "Accept: */*" \
            --header "Connection: keep-alive" \
            --form "op=upload_slice_data" \
            --form "session=$session" \
            --form "offset=$offset" \
            --form "filecontent=@$file")

        code=$(getjson "data['code']" "$resp")
        if [ $code -eq 0 ]; then
            debug "update slice $i success"
        else
            panic "$resp"
        fi

        offset=$(($offset + 4194304))
        rm $file
    done
}

uploadfinish() {
    token=$(gettoken)

    resp=$(curl -s --location --request POST "http://$wosserver/$appkey/$bucket/$filename" \
        --header "Authorization: $token" \
        --header "User-Agent: $version" \
        --header "Accept: */*" \
        --header "Connection: keep-alive" \
        --form "op=upload_slice_finish" \
        --form "session=$session" \
        --form "filesize=$size")

    code=$(getjson "data['code']" "$resp")
    if [ $code -eq 0 ]; then
        # info "upload success,url: $(getjson data['data']['url'] $resp) intranet_url: $(getjson data['data']['url'] $resp) access_url: $(getjson data['data']['url'] $resp)"
        echo $resp
    else
        panic "$resp"
    fi
}

getsha1() {
    if [ "$(uname)" == "Darwin" ]; then
        shasum $1 | awk '{print $1}'
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        sha1sum $1 | awk '{print $1}'
    # elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ];then
    #     debug "windows"
    else
        panic "unsupport OS"
    fi
}

# bashsupport disable=BP2001
precheck() {
    offset=0
    parts=""
    for i in $(seq $slicenum); do
        index=$(($i - 1))
        file=$(printf "$localfile.tmp.%.8d\n" $index)

        if [ ! -f $file ]; then
            panic "couldn't find file $file"
        fi

        sha1=$(getsha1 $file)
        slicelen=$(getsize "$file")
        parts="$parts,{\"offset\":$offset,\"datalen\":$slicelen,\"datasha\":\"$sha1\"}"
        offset=$(($offset + 4194304))
    done
    parts="[${parts:1}]"

    token=$(gettoken)
    resp=$(curl -s --location --request POST "http://$wosserver/$appkey/$bucket/$filename" \
        --header "Authorization: $token" \
        --header "User-Agent: $version" \
        --header "Accept: */*" \
        --header "Connection: keep-alive" \
        --form "op=upload_precheck" \
        --form "filesize=$size" \
        --form "ttl=$ttl" \
        --form "slice_size=4194304" \
        --form "uploadparts=$parts")
    debug "precheck resp:$resp"

    code=$(getjson "data['code']" "$resp")
    if [ $code -ne 0 ]; then
        panic "$resp"
    fi

    allhit=$(getjson "data['data']['allhit']" "$resp")
    if [ $allhit -eq 1 ]; then
        echo $resp
        exit 0
    fi

    session=$(getjson "data['data']['session']" "$resp")

    # len=$(echo "$resp" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data['data']['uploadparts']))")
    len=$(getjson "len(data['data']['uploadparts'])" "$resp")
    debug "slice num $len need upload"

    for i in $(seq $len); do
        index=$(($i - 1))
        # offset=$(echo "$resp" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data']['uploadparts'][$index]['offset'])")
        offset=$(getjson "data['data']['uploadparts'][$index]['offset']" $resp)
        needUpload="$needUpload $offset "
    done

    debug "need to upload offset array:$needUpload"
}

checkfile() {
    for i in $(seq $slicenum); do
        file=$(printf "$localfile.tmp.%.8d\n" $(($i - 1)))

        if [ -f $file ]; then
            panic "file $file has exist(wos-client need split $localfile to slice file)"
        fi
    done
}

uploadlarge() {
    slicenum=$(($size / 4194304))
    if [ $(($size % 4194304)) -gt 0 ]; then
        slicenum=$(($slicenum + 1))
    fi

    checkfile

    split -a 8 -b 4m -d $localfile $localfile.tmp.

    if [ $needprecheck -eq 1 ]; then
        precheck
    else
        uploadinit
    fi

    uploadslice

    uploadfinish
}

upload() {
    size=$(getsize "$localfile")
    if [ $size -le 4194304 ]; then
        uploadsmall
    else
        uploadlarge
    fi
}

delete() {
    token=$(gettoken)

    resp=$(curl -s --location --request POST "http://$wosserver/$appkey/$bucket/$filename" \
        --header "Authorization: $token" \
        --header "User-Agent: $version" \
        --header "Accept: */*" \
        --header "Connection: keep-alive" \
        --form "op=delete")

    code=$(getjson "data['code']" "$resp")
    if [ $code -eq 0 ]; then
        # info "upload success,url: $(getjson data['data']['url'] $resp) intranet_url: $(getjson data['data']['url'] $resp) access_url: $(getjson data['data']['url'] $resp)"
        echo $resp
    else
        panic "$resp"
    fi
}

setttl() {
    token=$(gettoken)

    resp=$(curl -s --location --request POST "http://$wosserver/$appkey/$bucket/$filename" \
        --header "Authorization: $token" \
        --header "User-Agent: $version" \
        --header "Accept: */*" \
        --header "Connection: keep-alive" \
        --form "op=set_ttl" \
        --form "ttl=$ttl")

    code=$(getjson "data['code']" "$resp")
    if [ $code -eq 0 ]; then
        # info "upload success,url: $(getjson data['data']['url'] $resp) intranet_url: $(getjson data['data']['url'] $resp) access_url: $(getjson data['data']['url'] $resp)"
        echo $resp
    else
        panic "$resp"
    fi
}

setfileread() {
    token=$(gettoken)

    resp=$(curl -s --location --request POST "http://$wosserver/$appkey/$bucket/$filename" \
        -u $appkey:$secret \
        --header "User-Agent: $version" \
        --header "Accept: */*" \
        --header "Connection: keep-alive" \
        --form "op=set_fileattr" \
        --form "private=$1")

    code=$(getjson "data['code']" "$resp")
    if [ $code -eq 0 ]; then
        # info "upload success,url: $(getjson data['data']['url'] $resp) intranet_url: $(getjson data['data']['url'] $resp) access_url: $(getjson data['data']['url'] $resp)"
        echo $resp
    else
        panic "$resp"
    fi
}

op=$(echo $other | awk '{print $1}')
case "$op" in
"upload")
    userinfo
    upload
    ;;
"delete")
    userinfo
    delete
    ;;
"get-token")
    userinfo
    gettoken
    ;;
"set-ttl")
    userinfo
    setttl
    ;;
"set-file-read")
    userinfo
    read=$(echo $other | awk '{print $2}')
    if [[ $read == "public" ]]; then
        attr=0
    elif [[ $read == "private" ]]; then
        attr=1
    else
        panic "unsupport $read"
    fi
    setfileread "$attr"
    ;;
*)
    panic "operation $op not allowed"
    help
    ;;
esac
