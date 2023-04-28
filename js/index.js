const cv = require("@techstark/opencv-js")
const ort = require("onnxruntime-web")


// for registering state proxy callbacks
const callbacks = {}
// internal state of the app
const state = new Proxy({}, {
    // intercept property set
    set: (obj, key, value) => {
        // change property only if there's a difference
        if (obj[key] == value) return true
        // log and change property
        console.log(`state.${key} = ${obj[key]} => ${value}`)
        obj[key] = value
        // invoke callbacks and return
        try {
            if (key in callbacks) callbacks[key].forEach(callback => callback(value))
        } catch (exc) {
            window.alert(`Not all combinations of options will work, some will fail, like this one - ${exc.toString()}`)
            throw exc
        }
        return true
    }
})


// generate simple load functions
function identity(value) {
    return value
}
function get_value(target) {
    return target.value
}
function get_checked(target) {
    return target.checked
}
function get_check_set_load(key, parse = identity, value = get_value, object = state) {
    function load(ev) {
        // simply check and set on object
        if (ev.target.reportValidity()) object[key] = parse(value(ev.target))
    }
    return load
}


// manual change dispatch
function change(el) {
    // simply dispatch with new event
    el.dispatchEvent(new Event("change"))
}


// plot histogram
function plot_hist(src_el, dst_el, rows = undefined, bins = 256, min = 0, max = 256) {
    // grab or read src
    const src = src_el instanceof cv.Mat ? src_el : cv.imread(src_el)
    // coalesce rows
    rows = (rows || undefined) ?? src.rows
    // need to pass src as vector
    const srcs = new cv.MatVector()
    srcs.push_back(src)
    // need to pass, not used
    const mask = new cv.Mat()
    // result histogram
    const hist = new cv.Mat()
    // result plot
    const dst = new cv.Mat.zeros(rows, bins, cv.CV_8UC3)
    try {
        // grab channels and calculate height of each histogram
        const channels = src.channels()
        const height = Math.trunc(rows / channels)
        // start with red and loop channels
        let color = new cv.Scalar(255, 0, 0)
        let channel = -1
        while (++channel < channels) {
            // calculate channel histogram and its bounds
            cv.calcHist(srcs, [channel], mask, hist, [bins], [min, max])
            const bounds = cv.minMaxLoc(hist, mask)
            // calculate histogram plot floor height
            const floor = height * channel + height
            // loop bins
            let bin = -1
            while (++bin < bins) {
                // calculate plot value from hist
                // note that hist contains raw pixel counts, as such normalize with max and scale to height
                const value = Math.trunc(hist.data32F[bin] / bounds.maxVal * (height - 1))
                // don't draw points with no value
                if (!value) continue
                // draw a vertical line from bot to top
                const bot = new cv.Point(bin, floor)
                const top = new cv.Point(bin, floor - value)
                cv.line(dst, bot, top, color, 1)
            }
            // rotate color
            color = new cv.Scalar(color[2], color[0], color[1])
        }
        // show histogram
        cv.imshow(dst_el, dst)
    } finally {
        // deallocate all
        dst.delete()
        hist.delete()
        mask.delete()
        srcs.delete()
        if (!(src_el instanceof cv.Mat)) src.delete()
    }
}


// execute callback per channel
function exec_channels(src, dst, callback) {
    // grab and check channels
    const channels = src.channels()
    if (channels == 1) return callback(src, dst)
    // got multiple channels, initialize vectors
    const srcs = new cv.MatVector()
    const dsts = new cv.MatVector()
    try {
        // split image and loop channels
        cv.split(src, srcs)
        let channel = -1
        while (++channel < channels) {
            // grab src and dst for channel
            const _src = srcs.get(channel)
            const _dst = new cv.Mat()
            try {
                // execute callback and save dst
                callback(_src, _dst)
                dsts.push_back(_dst)
            } finally {
                // seems that vectors use protective copy, deallocate the ones we got
                _src.delete()
                _dst.delete()
            }
        }
        // merge channels back
        if (dst) cv.merge(dsts, dst)
    } finally {
        // deallocate vectors
        srcs.delete()
        dsts.delete()
    }
}


// execute image processing stage
function process_img(dst_el, hist_el, check, check_hist, previous, next, process) {
    // check if we have previous stage
    if (!state[previous]) return
    // allocate next if we process, else clone
    let mat_next = state[check] ? new cv.Mat() : state[previous].clone()
    try {
        mat_next = process(state[previous], mat_next)
        // show next and plot histogram
        cv.imshow(dst_el, mat_next)
        if (state[check_hist]) plot_hist(mat_next, hist_el)
    } catch (exc) {
        // failed, deallocate next
        mat_next.delete()
        throw exc
    }
    // deallocate previous and set next
    if (state[next]) state[next].delete()
    state[next] = mat_next
}


// generate histogram
function process_hist(hist_el, hist, mat) {
    // hide based on value and reload if histogram is required
    hist_el.hidden = !hist
    if (hist && mat) plot_hist(mat, hist_el)
}





// stage 1 - original input





// load image and video
function load_src(src, type) {
    // src is already prepared, simply check type and set on internal state
    if (type.startsWith("image")) state.img_src = URL.createObjectURL(src)
    else if (type.startsWith("video")) state.video_src = URL.createObjectURL(src)
}
function set_img_src(img_src, img_el, video_el, input_playback_rate_el, input_loop_el) {
    // got img, pause and hide video el
    video_el.pause()
    video_el.hidden = true
    // disable video specific controls
    input_playback_rate_el.disabled = true
    input_loop_el.disabled = true
    // set and show img el
    img_el.src = img_src
    img_el.hidden = false
}
function set_video_src(video_src, img_el, video_el, input_playback_rate_el, input_loop_el) {
    // got video, hide img el
    img_el.hidden = true
    // enable video specific controls
    input_playback_rate_el.disabled = false
    input_loop_el.disabled = false
    // set and show video el
    video_el.src = video_src
    video_el.hidden = false
    // seems that we also need to reset video specific options
    video_el.playbackRate = state.playback_rate
    video_el.loop = state.loop
}


// load from file and url
function load_file(input_file_el) {
    // check el and files and load the first
    if (input_file_el.reportValidity() && input_file_el.files.length) load_src(input_file_el.files[0], input_file_el.files[0].type)
}
async function load_url(input_url_el, proxies) {
    // check el
    if (!input_url_el.reportValidity()) return
    // valid, grab and fetch url
    const url = input_url_el.value
    let response
    try {
        response = await fetch(url)
    } catch {
        // fetch fail, assume cors
        // shuffle proxies for load balance and loop
        proxies.sort(() => .5 - Math.random())
        let proxy_index = 0
        while (proxy_index < proxies.length) {
            try {
                // fetch from proxy
                response = await fetch(`${proxies[proxy_index++]}${url}`)
                // success, stop
                break
            } catch {
                // fail, next proxy
                continue
            }
        }
    }
    // check if we got response and load blob if so
    if (response) load_src(await response.blob(), response.headers.get("Content-Type"))
}


// load from drag & drop
function load_drop(ev, input_file_el, input_url_el, submit_fetch_el) {
    // stop browser from default and check if its files or urls
    ev.preventDefault()
    if (ev.dataTransfer.files.length) {
        // files, load through file el
        input_file_el.files = ev.dataTransfer.files
        // can't change value, need to fire change event manually
        change(input_file_el)
    } else if (ev.dataTransfer.items.length) ev.dataTransfer.items[0].getAsString(url => {
        // urls, load through url el
        input_url_el.value = url
        submit_fetch_el.click()
    })
}





// stage 2 - initial processing





// remember and load initial image
function save_load(load) {
    // store and execute load function
    state.load = load
    load()
}
function load_img(
    src_el, dst_el, canvas_el, canvas_ctx, width, height, hist_el, initial_hist_el,
    output_original_width_el, output_original_height_el, output_actual_fps_el) {
    // coalesce width and height
    canvas_el.width = state.width
    canvas_el.width ||= width
    canvas_el.height = state.height
    canvas_el.height ||= height
    // resize and read new
    canvas_ctx.drawImage(src_el, 0, 0, canvas_el.width, canvas_el.height)
    let mat_initial = cv.imread(canvas_el)
    try {
        // plot original histogram and convert color space
        if (state.hist) plot_hist(mat_initial, hist_el, src_el.height)
        if (state.color_space) cv.cvtColor(mat_initial, mat_initial, cv[state.color_space])
        // extract channel
        if (state.channel) {
            // initialize vector
            const channels = new cv.MatVector()
            try {
                // split and grab channel
                cv.split(mat_initial, channels)
                const channel = channels.get(state.channel - 1)
                // deallocate old and swap
                mat_initial.delete()
                mat_initial = channel
            } finally {
                // deallocate vector
                channels.delete()
            }
        }
        // show new and plot histogram
        cv.imshow(dst_el, mat_initial)
        if (state.initial_hist) plot_hist(mat_initial, initial_hist_el)
    } catch (exc) {
        // failed, deallocate new
        mat_initial.delete()
        throw exc
    }
    // show original size
    output_original_width_el.innerHTML = src_el.videoWidth ?? src_el.width
    output_original_height_el.innerHTML = src_el.videoHeight ?? src_el.height
    output_actual_fps_el.innerHTML = 0
    // deallocate previous and set new
    if (state.mat_initial) state.mat_initial.delete()
    state.mat_initial = mat_initial
}


// process video at given fps
function load_play(video_el) {
    // simply set on internal state
    state.play = !video_el.paused && !video_el.ended
}
function set_play(play, output_actual_fps_el) {
    if (!play || state.playing) return
    // need to play and not playing, define and schedule callback
    function callback() {
        try {
            // mark start time
            const start = performance.now()
            // reload
            state.load()
            // calculate delta and show actual fps
            const delta = performance.now() - start
            output_actual_fps_el.innerHTML = 1000 / delta
            // calculate free time until next frame
            const free = state.fps > 0 ? 1000 / state.fps - delta : 0
            // schedule next callback if we're still playing
            state.playing = state.play ? setTimeout(callback, Math.max(0, free)) : undefined
        } catch (exc) {
            // exc, stop playing and rethrow
            state.playing = undefined
            throw exc
        }
    }
    state.playing = setTimeout(callback)
}





// stage 3 - blur





// blur image
function blur_img(mat_previous, mat_next) {
    // blur based on type
    switch (state.blur) {
        case "blur":
            cv.blur(mat_previous, mat_next, new cv.Size(state.blur_kernel_width, state.blur_kernel_height))
            break
        case "GaussianBlur":
            cv.GaussianBlur(mat_previous, mat_next, new cv.Size(state.blur_kernel_width, state.blur_kernel_height), state.sigma_x, state.sigma_y)
            break
        case "medianBlur":
            cv.medianBlur(mat_previous, mat_next, state.blur_kernel_size)
            break
        case "bilateralFilter":
            exec_channels(mat_previous, mat_next, (src, dst) => cv.bilateralFilter(src, dst, state.blur_diameter, state.sigma_color, state.sigma_space))
            break
    }
    return mat_next
}


// switch inputs
function set_blur(
    blur,
    input_blur_kernel_width_el, input_blur_kernel_height_el, input_blur_kernel_size_el, input_blur_diameter_el,
    input_sigma_x_el, input_sigma_y_el, input_sigma_color_el, input_sigma_space_el) {
    input_blur_kernel_width_el.disabled = true
    input_blur_kernel_height_el.disabled = true
    input_blur_kernel_size_el.disabled = true
    input_blur_diameter_el.disabled = true
    input_sigma_x_el.disabled = true
    input_sigma_y_el.disabled = true
    input_sigma_color_el.disabled = true
    input_sigma_space_el.disabled = true
    switch (blur) {
        case "blur":
            input_blur_kernel_width_el.disabled = false
            input_blur_kernel_height_el.disabled = false
            break
        case "GaussianBlur":
            input_blur_kernel_width_el.disabled = false
            input_blur_kernel_height_el.disabled = false
            input_sigma_x_el.disabled = false
            input_sigma_y_el.disabled = false
            break
        case "medianBlur":
            input_blur_kernel_size_el.disabled = false
            break
        case "bilateralFilter":
            input_blur_diameter_el.disabled = false
            input_sigma_color_el.disabled = false
            input_sigma_space_el.disabled = false
            break
    }
}





// stage 4 - histogram equalization





// equalize image
function equalize_img(mat_previous, mat_next) {
    if (state.equalization) {
        if (state.equalization == "CLAHE") {
            const clahe = new cv.CLAHE(state.equalization_clip, new cv.Size(state.equalization_columns, state.equalization_rows))
            try {
                exec_channels(mat_previous, mat_next, (src, dst) => clahe.apply(src, dst))
            } finally {
                clahe.delete()
            }
        } else {
            exec_channels(mat_previous, mat_next, cv.equalizeHist)
        }
    }
    return mat_next
}


// switch inputs
function set_equalization(
    equalization,
    input_equalization_clip_el, input_equalization_rows_el, input_equalization_columns_el) {
    const disabled = equalization != "CLAHE"
    input_equalization_clip_el.disabled = disabled
    input_equalization_rows_el.disabled = disabled
    input_equalization_columns_el.disabled = disabled
}





// stage 5 - threshold





// threshold image
function threshold_img(mat_previous, mat_next) {
    if (state.threshold) {
        if (state.adaptive_threshold) {
            exec_channels(mat_previous, mat_next, (src, dst) => cv.adaptiveThreshold(
                src, dst, state.threshold_max,
                cv[state.adaptive_threshold], cv[state.threshold],
                state.threshold_block, state.threshold_constant))
        } else if (state.optimal_threshold) {
            exec_channels(mat_previous, mat_next, (src, dst) => cv.threshold(
                src, dst, state.threshold_value, state.threshold_max,
                cv[state.threshold] | cv[state.optimal_threshold]))
        } else {
            cv.threshold(
                mat_previous, mat_next, state.threshold_value, state.threshold_max,
                cv[state.threshold])
        }
    }
    return mat_next
}


// switch inputs
function set_threshold(
    threshold,
    input_threshold_value_el, input_threshold_max_el,
    input_optimal_threshold_el, input_adaptive_threshold_el,
    input_threshold_block_el, input_threshold_constant_el) {
    input_threshold_value_el.disabled = true
    input_threshold_max_el.disabled = true
    input_optimal_threshold_el.disabled = true
    input_adaptive_threshold_el.disabled = true
    input_threshold_block_el.disabled = true
    input_threshold_constant_el.disabled = true
    switch (threshold) {
        case "THRESH_BINARY":
        case "THRESH_BINARY_INV":
            input_threshold_value_el.disabled = false
            input_threshold_max_el.disabled = false
            break
        case "THRESH_TRUNC":
        case "THRESH_TOZERO":
        case "THRESH_TOZERO_INV":
            input_threshold_value_el.disabled = false
            break
        default:
            return
    }
    if (state.optimal_threshold) {
        input_threshold_value_el.disabled = true
    } else {
        input_adaptive_threshold_el.disabled = false
    }
    if (state.adaptive_threshold) {
        input_threshold_value_el.disabled = true
        input_threshold_block_el.disabled = false
        input_threshold_constant_el.disabled = false
    } else {
        input_optimal_threshold_el.disabled = false
    }
}





// stage 6 - canny & contours





// canny image
function canny_img(mat_previous, mat_next) {
    if (state.canny) exec_channels(
        mat_previous, mat_next,
        (src, dst) => cv.Canny(src, dst, state.canny_min, state.canny_max, state.canny_sobel, state.canny_l2))
    if (state.contours_mode) {
        // allocate zeros, note that cv.drawContours doesn't initialize
        const mat_contours = cv.Mat.zeros(mat_next.rows, mat_next.cols, cv.CV_8UC3)
        try {
            // process each channel, note that there is no dst due to cv.findContours requiring cv.CV_8UC1 src while cv.drawContours cv.CV_8UC3 dst
            exec_channels(mat_next, undefined, src => {
                // allocate contours and hierarchy, note that hierarchy is effectively unused
                const contours = new cv.MatVector()
                const hierarchy = new cv.Mat()
                try {
                    // find, loop and draw each contour with random color
                    cv.findContours(src, contours, hierarchy, cv.RETR_LIST, cv[state.contours_mode])
                    let contour = contours.size()
                    while (contour--) cv.drawContours(
                        mat_contours, contours, contour,
                        new cv.Scalar(Math.floor(Math.random() * 255), Math.floor(Math.random() * 255), Math.floor(Math.random() * 255)),
                        1, cv.LINE_8, hierarchy, 255)
                } finally {
                    // deallocate contours and hierarchy
                    contours.delete()
                    hierarchy.delete()
                }
            })
            // swap mat_contours with mat_next
            mat_next.delete()
            mat_next = mat_contours
        } catch (exc) {
            // failed, deallocate mat_contours
            mat_contours.delete()
            throw exc
        }
    }
    return mat_next
}


// switch inputs
function set_canny(
    canny,
    input_canny_min_el, input_canny_max_el,
    input_canny_sobel_el, input_canny_l2_el) {
    const disabled = !canny
    input_canny_min_el.disabled = disabled
    input_canny_max_el.disabled = disabled
    input_canny_sobel_el.disabled = disabled
    input_canny_l2_el.disabled = disabled
}





// stage 7 - hough lines





// hough image
function hough_img(mat_previous, mat_next) {
    if (state.hough) exec_channels(mat_previous, mat_next, (src, dst) => {
        // initialize dst and color since we draw manually
        dst.create(src.rows, src.cols, cv.CV_8UC1)
        dst.setTo(new cv.Scalar(0))
        const color = new cv.Scalar(255)
        // allocate lines and check hough
        const lines = new cv.Mat()
        try {
            if (state.hough == "HoughLines") {
                // standard, execute and loop lines
                cv.HoughLines(
                    src, lines, state.hough_rho, state.hough_theta * Math.PI / 180, state.hough_threshold,
                    state.hough_srn, state.hough_stn, state.hough_theta_min, state.hough_theta_max)
                let line = lines.rows * 2
                while ((line -= 2) >= 0) {
                    // draw line, see https://docs.opencv.org/3.4/d3/de6/tutorial_js_houghlines.html
                    const rho = lines.data32F[line]
                    const theta = lines.data32F[line + 1]
                    const a = Math.cos(theta)
                    const b = Math.sin(theta)
                    const x = a * rho
                    const y = b * rho
                    cv.line(
                        dst, new cv.Point(Math.floor(x - 1000 * b), Math.floor(y + 1000 * a)),
                        new cv.Point(Math.floor(x + 1000 * b), Math.floor(y - 1000 * a)), color)
                }
            } else {
                // probabilistic, execute, loop and draw lines
                cv.HoughLinesP(
                    src, lines, state.hough_rho, state.hough_theta * Math.PI / 180, state.hough_threshold,
                    state.hough_min_length, state.hough_max_gap)
                let line = lines.rows * 4
                while ((line -= 4) >= 0) cv.line(
                    dst, new cv.Point(lines.data32S[line], lines.data32S[line + 1]),
                    new cv.Point(lines.data32S[line + 2], lines.data32S[line + 3]), color)
            }
        } finally {
            // deallocate lines
            lines.delete()
        }
    })
    return mat_next
}


// switch inputs
function set_hough(
    hough,
    input_hough_rho_el, input_hough_theta_el, input_hough_threshold_el,
    input_hough_srn_el, input_hough_stn_el,
    input_hough_theta_min_el, input_hough_theta_max_el,
    input_hough_min_length_el, input_hough_max_gap_el) {
    input_hough_rho_el.disabled = true
    input_hough_theta_el.disabled = true
    input_hough_threshold_el.disabled = true
    input_hough_srn_el.disabled = true
    input_hough_stn_el.disabled = true
    input_hough_theta_min_el.disabled = true
    input_hough_theta_max_el.disabled = true
    input_hough_min_length_el.disabled = true
    input_hough_max_gap_el.disabled = true
    switch (hough) {
        case "HoughLines":
            input_hough_rho_el.disabled = false
            input_hough_theta_el.disabled = false
            input_hough_threshold_el.disabled = false
            input_hough_srn_el.disabled = false
            input_hough_stn_el.disabled = false
            input_hough_theta_min_el.disabled = false
            input_hough_theta_max_el.disabled = false
            break
        case "HoughLinesP":
            input_hough_rho_el.disabled = false
            input_hough_theta_el.disabled = false
            input_hough_threshold_el.disabled = false
            input_hough_min_length_el.disabled = false
            input_hough_max_gap_el.disabled = false
            break
    }
}





// initialization





// main entry point
function main() {
    //
    // stage 1 - original input
    //
    // grab IO UI elements
    const output_img_el = document.getElementById("output-img")
    const output_video_el = document.getElementById("output-video")
    const output_hist_el = document.getElementById("output-hist")
    const input_file_el = document.getElementById("input-file")
    const submit_fetch_el = document.getElementById("submit-fetch")
    const input_url_el = document.getElementById("input-url")
    const input_playback_rate_el = document.getElementById("input-playback-rate")
    const input_loop_el = document.getElementById("input-loop")
    const input_hist_el = document.getElementById("input-hist")
    const output_original_width_el = document.getElementById("output-original-width")
    const output_original_height_el = document.getElementById("output-original-height")
    const output_actual_fps_el = document.getElementById("output-actual-fps")
    // grab temp canvas and its context
    const temp_canvas_el = document.getElementById("temp-canvas")
    const temp_canvas_ctx = temp_canvas_el.getContext("2d")
    // proxies in case of cors problems
    const proxies = [
        // "https://cors-proxy.htmldriven.com/?url=",
        "https://corsproxy.io/?",
        // "https://crossorigin.me/",
        // "https://api.allorigins.win/raw?url=",
    ]
    const callback_load_img = () => load_img(
        output_img_el, output_initial_el,
        temp_canvas_el, temp_canvas_ctx,
        output_img_el.width, output_img_el.height,
        output_hist_el, output_initial_hist_el,
        output_original_width_el, output_original_height_el, output_actual_fps_el)
    const callback_load_video = () => load_img(
        output_video_el, output_initial_el,
        temp_canvas_el, temp_canvas_ctx,
        output_video_el.clientWidth, output_video_el.clientHeight,
        output_hist_el, output_initial_hist_el,
        output_original_width_el, output_original_height_el, output_actual_fps_el)
    // register callbacks to react to internal state changes
    callbacks.play = [play => set_play(play, output_actual_fps_el)]
    callbacks.playback_rate = [playback_rate => output_video_el.playbackRate = playback_rate]
    callbacks.loop = [loop => output_video_el.loop = loop]
    callbacks.hist = [hist => process_hist(output_hist_el, hist, temp_canvas_el)]
    // react to img/video src change fired by multiple methods
    callbacks.img_src = [img_src => set_img_src(img_src, output_img_el, output_video_el, input_playback_rate_el, input_loop_el)]
    callbacks.video_src = [video_src => set_video_src(video_src, output_img_el, output_video_el, input_playback_rate_el, input_loop_el)]
    // register callbacks to change internal state on element changes
    output_video_el.onplay = () => load_play(output_video_el)
    output_video_el.onpause = () => load_play(output_video_el)
    input_file_el.onchange = () => load_file(input_file_el)
    submit_fetch_el.onclick = () => load_url(input_url_el, proxies)
    input_playback_rate_el.onchange = get_check_set_load("playback_rate", parseFloat)
    input_loop_el.onchange = get_check_set_load("loop", identity, get_checked)
    input_hist_el.onchange = get_check_set_load("hist", identity, get_checked)
    // react to drag & drop on the entire document
    document.ondragover = ev => ev.preventDefault()
    document.ondrop = ev => load_drop(ev, input_file_el, input_url_el, submit_fetch_el)
    // react to the original src load
    // note that we remember which one we did last so that we can repeat it when stage config changes
    output_img_el.onload = () => save_load(callback_load_img)
    output_video_el.onloadeddata = () => save_load(callback_load_video)
    // also react to the user seeking to a new time in video
    output_video_el.onseeked = callback_load_video
    // fire element changes to initialize and sync internal state
    load_play(output_video_el)
    change(input_file_el)
    submit_fetch_el.click()
    change(input_playback_rate_el)
    change(input_loop_el)
    change(input_hist_el)
    //
    // stage 2 - initial processing
    //
    const output_initial_el = document.getElementById("output-initial")
    const output_initial_hist_el = document.getElementById("output-initial-hist")
    const input_width_el = document.getElementById("input-width")
    const input_height_el = document.getElementById("input-height")
    const input_color_space_el = document.getElementById("input-color-space")
    const input_channel_el = document.getElementById("input-channel")
    const input_fps_el = document.getElementById("input-fps")
    const input_initial_hist_el = document.getElementById("input-initial-hist")
    const callback_load = () => state.load?.()
    callbacks.width = [callback_load]
    callbacks.height = [callback_load]
    callbacks.color_space = [callback_load]
    callbacks.channel = [callback_load]
    callbacks.initial_hist = [initial_hist => process_hist(output_initial_hist_el, initial_hist, state.mat_initial)]
    // enable fps only for video
    // note that this is monitered in set_play
    callbacks.img_src.push(() => input_fps_el.disabled = true)
    callbacks.video_src.push(() => input_fps_el.disabled = false)
    input_width_el.onchange = get_check_set_load("width", parseInt)
    input_height_el.onchange = get_check_set_load("height", parseInt)
    input_color_space_el.onchange = get_check_set_load("color_space")
    input_channel_el.onchange = get_check_set_load("channel", parseInt)
    input_fps_el.onchange = get_check_set_load("fps", parseInt)
    input_initial_hist_el.onchange = get_check_set_load("initial_hist", identity, get_checked)
    // dynamic options for color space based on whats available on cv
    Object.keys(cv).filter(key => key.startsWith("COLOR_")).forEach(key => {
        const input_color_space_option_el = document.createElement("option")
        input_color_space_option_el.value = key
        input_color_space_option_el.innerHTML = key.slice(6)
        input_color_space_el.appendChild(input_color_space_option_el)
    })
    change(input_width_el)
    change(input_height_el)
    change(input_color_space_el)
    change(input_channel_el)
    change(input_fps_el)
    change(input_initial_hist_el)
    //
    // stage 3 - blur
    //
    const output_blur_el = document.getElementById("output-blur")
    const output_blur_hist_el = document.getElementById("output-blur-hist")
    const input_blur_el = document.getElementById("input-blur")
    const input_blur_kernel_width_el = document.getElementById("input-blur-kernel-width")
    const input_blur_kernel_height_el = document.getElementById("input-blur-kernel-height")
    const input_blur_kernel_size_el = document.getElementById("input-blur-kernel-size")
    const input_blur_diameter_el = document.getElementById("input-blur-diameter")
    const input_sigma_x_el = document.getElementById("input-sigma-x")
    const input_sigma_y_el = document.getElementById("input-sigma-y")
    const input_sigma_color_el = document.getElementById("input-sigma-color")
    const input_sigma_space_el = document.getElementById("input-sigma-space")
    const input_blur_hist_el = document.getElementById("input-blur-hist")
    const callback_blur = () => process_img(
        output_blur_el, output_blur_hist_el,
        "blur", "blur_hist", "mat_initial", "mat_blur",
        blur_img)
    callbacks.mat_initial = [callback_blur]
    callbacks.blur = [blur => set_blur(
        blur,
        input_blur_kernel_width_el, input_blur_kernel_height_el, input_blur_kernel_size_el, input_blur_diameter_el,
        input_sigma_x_el, input_sigma_y_el, input_sigma_color_el, input_sigma_space_el), callback_blur]
    callbacks.blur_kernel_width = [callback_blur]
    callbacks.blur_kernel_height = [callback_blur]
    callbacks.blur_kernel_size = [callback_blur]
    callbacks.blur_diameter = [callback_blur]
    callbacks.sigma_x = [callback_blur]
    callbacks.sigma_y = [callback_blur]
    callbacks.sigma_color = [callback_blur]
    callbacks.sigma_space = [callback_blur]
    callbacks.blur_hist = [blur_hist => process_hist(output_blur_hist_el, blur_hist, state.mat_blur)]
    input_blur_el.onchange = get_check_set_load("blur")
    input_blur_kernel_width_el.onchange = get_check_set_load("blur_kernel_width", parseInt)
    input_blur_kernel_height_el.onchange = get_check_set_load("blur_kernel_height", parseInt)
    input_blur_kernel_size_el.onchange = get_check_set_load("blur_kernel_size", parseInt)
    input_blur_diameter_el.onchange = get_check_set_load("blur_diameter", parseInt)
    input_sigma_x_el.onchange = get_check_set_load("sigma_x", parseFloat)
    input_sigma_y_el.onchange = get_check_set_load("sigma_y", parseFloat)
    input_sigma_color_el.onchange = get_check_set_load("sigma_color", parseInt)
    input_sigma_space_el.onchange = get_check_set_load("sigma_space", parseInt)
    input_blur_hist_el.onchange = get_check_set_load("blur_hist", identity, get_checked)
    change(input_blur_el)
    change(input_blur_kernel_width_el)
    change(input_blur_kernel_height_el)
    change(input_blur_kernel_size_el)
    change(input_blur_diameter_el)
    change(input_sigma_x_el)
    change(input_sigma_y_el)
    change(input_sigma_color_el)
    change(input_sigma_space_el)
    change(input_blur_hist_el)
    //
    // stage 4 - histogram equalization
    //
    const output_equalization_el = document.getElementById("output-equalization")
    const output_equalization_hist_el = document.getElementById("output-equalization-hist")
    const input_equalization_el = document.getElementById("input-equalization")
    const input_equalization_clip_el = document.getElementById("input-equalization-clip")
    const input_equalization_rows_el = document.getElementById("input-equalization-rows")
    const input_equalization_columns_el = document.getElementById("input-equalization-columns")
    const input_equalization_hist_el = document.getElementById("input-equalization-hist")
    const callback_equalization = () => process_img(
        output_equalization_el, output_equalization_hist_el,
        "equalization", "equalization_hist", "mat_blur", "mat_equalization",
        equalize_img)
    callbacks.mat_blur = [callback_equalization]
    callbacks.equalization = [equalization => set_equalization(
        equalization,
        input_equalization_clip_el, input_equalization_rows_el, input_equalization_columns_el), callback_equalization]
    callbacks.equalization_clip = [callback_equalization]
    callbacks.equalization_rows = [callback_equalization]
    callbacks.equalization_columns = [callback_equalization]
    callbacks.equalization_hist = [equalization_hist => process_hist(output_equalization_hist_el, equalization_hist, state.mat_equalization)]
    input_equalization_el.onchange = get_check_set_load("equalization")
    input_equalization_clip_el.onchange = get_check_set_load("equalization_clip", parseInt)
    input_equalization_rows_el.onchange = get_check_set_load("equalization_rows", parseInt)
    input_equalization_columns_el.onchange = get_check_set_load("equalization_columns", parseInt)
    input_equalization_hist_el.onchange = get_check_set_load("equalization_hist", identity, get_checked)
    change(input_equalization_el)
    change(input_equalization_clip_el)
    change(input_equalization_rows_el)
    change(input_equalization_columns_el)
    change(input_equalization_hist_el)
    //
    // stage 5 - threshold
    //
    const output_threshold_el = document.getElementById("output-threshold")
    const output_threshold_hist_el = document.getElementById("output-threshold-hist")
    const input_threshold_el = document.getElementById("input-threshold")
    const input_threshold_value_el = document.getElementById("input-threshold-value")
    const input_threshold_max_el = document.getElementById("input-threshold-max")
    const input_optimal_threshold_el = document.getElementById("input-optimal-threshold")
    const input_adaptive_threshold_el = document.getElementById("input-adaptive-threshold")
    const input_threshold_block_el = document.getElementById("input-threshold-block")
    const input_threshold_constant_el = document.getElementById("input-threshold-constant")
    const input_threshold_hist_el = document.getElementById("input-threshold-hist")
    const callback_threshold = () => process_img(
        output_threshold_el, output_threshold_hist_el,
        "threshold", "threshold_hist", "mat_equalization", "mat_threshold",
        threshold_img)
    const callback_set_threshold = threshold => set_threshold(
        threshold,
        input_threshold_value_el, input_threshold_max_el,
        input_optimal_threshold_el, input_adaptive_threshold_el,
        input_threshold_block_el, input_threshold_constant_el)
    callbacks.mat_equalization = [callback_threshold]
    callbacks.threshold = [callback_set_threshold, callback_threshold]
    callbacks.threshold_value = [callback_threshold]
    callbacks.threshold_max = [callback_threshold]
    callbacks.optimal_threshold = [() => callback_set_threshold(state.threshold), callback_threshold]
    callbacks.adaptive_threshold = [() => callback_set_threshold(state.threshold), callback_threshold]
    callbacks.threshold_block = [callback_threshold]
    callbacks.threshold_constant = [callback_threshold]
    callbacks.threshold_hist = [threshold_hist => process_hist(output_threshold_hist_el, threshold_hist, state.mat_threshold)]
    input_threshold_el.onchange = get_check_set_load("threshold")
    input_threshold_value_el.onchange = get_check_set_load("threshold_value", parseInt)
    input_threshold_max_el.onchange = get_check_set_load("threshold_max", parseInt)
    input_optimal_threshold_el.onchange = get_check_set_load("optimal_threshold")
    input_adaptive_threshold_el.onchange = get_check_set_load("adaptive_threshold")
    input_threshold_block_el.onchange = get_check_set_load("threshold_block", parseInt)
    input_threshold_constant_el.onchange = get_check_set_load("threshold_constant", parseInt)
    input_threshold_hist_el.onchange = get_check_set_load("threshold_hist", identity, get_checked)
    change(input_threshold_el)
    change(input_threshold_value_el)
    change(input_threshold_max_el)
    change(input_optimal_threshold_el)
    change(input_adaptive_threshold_el)
    change(input_threshold_block_el)
    change(input_threshold_constant_el)
    change(input_threshold_hist_el)
    //
    // stage 6 - canny & contours
    //
    const output_canny_el = document.getElementById("output-canny")
    const output_canny_hist_el = document.getElementById("output-canny-hist")
    const input_canny_el = document.getElementById("input-canny")
    const input_canny_min_el = document.getElementById("input-canny-min")
    const input_canny_max_el = document.getElementById("input-canny-max")
    const input_canny_sobel_el = document.getElementById("input-canny-sobel")
    const input_canny_l2_el = document.getElementById("input-canny-l2")
    const input_contours_mode_el = document.getElementById("input-contours-mode")
    const input_canny_hist_el = document.getElementById("input-canny-hist")
    const callback_canny = () => process_img(
        output_canny_el, output_canny_hist_el,
        "canny", "canny_hist", "mat_threshold", "mat_canny",
        canny_img)
    callbacks.mat_threshold = [callback_canny]
    callbacks.canny = [canny => set_canny(
        canny,
        input_canny_min_el, input_canny_max_el,
        input_canny_sobel_el, input_canny_l2_el), callback_canny]
    callbacks.canny_min = [callback_canny]
    callbacks.canny_max = [callback_canny]
    callbacks.canny_sobel = [callback_canny]
    callbacks.canny_l2 = [callback_canny]
    callbacks.contours_mode = [callback_canny]
    callbacks.canny_hist = [canny_hist => process_hist(output_canny_hist_el, canny_hist, state.mat_canny)]
    input_canny_el.onchange = get_check_set_load("canny", identity, get_checked)
    input_canny_min_el.onchange = get_check_set_load("canny_min", parseInt)
    input_canny_max_el.onchange = get_check_set_load("canny_max", parseInt)
    input_canny_sobel_el.onchange = get_check_set_load("canny_sobel", parseInt)
    input_canny_l2_el.onchange = get_check_set_load("canny_l2", identity, get_checked)
    input_contours_mode_el.onchange = get_check_set_load("contours_mode")
    input_canny_hist_el.onchange = get_check_set_load("canny_hist", identity, get_checked)
    change(input_canny_el)
    change(input_canny_min_el)
    change(input_canny_max_el)
    change(input_canny_sobel_el)
    change(input_canny_l2_el)
    change(input_contours_mode_el)
    change(input_canny_hist_el)
    //
    // stage 7 - hough lines
    //
    const output_hough_el = document.getElementById("output-hough")
    const output_hough_hist_el = document.getElementById("output-hough-hist")
    const input_hough_el = document.getElementById("input-hough")
    const input_hough_rho_el = document.getElementById("input-hough-rho")
    const input_hough_theta_el = document.getElementById("input-hough-theta")
    const input_hough_threshold_el = document.getElementById("input-hough-threshold")
    const input_hough_srn_el = document.getElementById("input-hough-srn")
    const input_hough_stn_el = document.getElementById("input-hough-stn")
    const input_hough_theta_min_el = document.getElementById("input-hough-theta-min")
    const input_hough_theta_max_el = document.getElementById("input-hough-theta-max")
    const input_hough_min_length_el = document.getElementById("input-hough-min-length")
    const input_hough_max_gap_el = document.getElementById("input-hough-max-gap")
    const input_hough_hist_el = document.getElementById("input-hough-hist")
    const callback_hough = () => process_img(
        output_hough_el, output_hough_hist_el,
        "hough", "hough_hist", "mat_canny", "mat_hough",
        hough_img)
    callbacks.mat_canny = [callback_hough]
    callbacks.hough = [hough => set_hough(
        hough,
        input_hough_rho_el, input_hough_theta_el, input_hough_threshold_el,
        input_hough_srn_el, input_hough_stn_el,
        input_hough_theta_min_el, input_hough_theta_max_el,
        input_hough_min_length_el, input_hough_max_gap_el), callback_hough]
    callbacks.hough_rho = [callback_hough]
    callbacks.hough_theta = [callback_hough]
    callbacks.hough_threshold = [callback_hough]
    callbacks.hough_srn = [callback_hough]
    callbacks.hough_stn = [callback_hough]
    callbacks.hough_theta_min = [callback_hough]
    callbacks.hough_theta_max = [callback_hough]
    callbacks.hough_min_length = [callback_hough]
    callbacks.hough_max_gap = [callback_hough]
    callbacks.hough_hist = [hough_hist => process_hist(output_hough_hist_el, hough_hist, state.mat_hough)]
    input_hough_el.onchange = get_check_set_load("hough")
    input_hough_rho_el.onchange = get_check_set_load("hough_rho", parseInt)
    input_hough_theta_el.onchange = get_check_set_load("hough_theta", parseInt)
    input_hough_threshold_el.onchange = get_check_set_load("hough_threshold", parseInt)
    input_hough_srn_el.onchange = get_check_set_load("hough_srn", parseInt)
    input_hough_stn_el.onchange = get_check_set_load("hough_stn", parseInt)
    input_hough_theta_min_el.onchange = get_check_set_load("hough_theta_min", parseFloat)
    input_hough_theta_max_el.onchange = get_check_set_load("hough_theta_max", parseFloat)
    input_hough_min_length_el.onchange = get_check_set_load("hough_min_length", parseInt)
    input_hough_max_gap_el.onchange = get_check_set_load("hough_max_gap", parseInt)
    input_hough_hist_el.onchange = get_check_set_load("hough_hist", identity, get_checked)
    change(input_hough_el)
    change(input_hough_rho_el)
    change(input_hough_theta_el)
    change(input_hough_threshold_el)
    change(input_hough_srn_el)
    change(input_hough_stn_el)
    change(input_hough_theta_min_el)
    change(input_hough_theta_max_el)
    change(input_hough_min_length_el)
    change(input_hough_max_gap_el)
    change(input_hough_hist_el)
}


// seems that opencv does dynamic initialization, register main to be executed after that
cv.onRuntimeInitialized = main
